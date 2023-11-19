# Project 5 - Traffic

Developing a neural network involves various aspects that can significantly impact its performance on the given task. The objective of this project was to create a neural network capable of identifying traffic signs. The implementation was done in Python using TensorFlow, scikit-learn, and OpenCV-Python libraries, and the German Traffic Sign Recognition Benchmark (GTSRB) dataset was used as the training data.

## Initial Model

During the tests, the first step was to build a simple network, thus the initial model was constructed with the following layers:

- Conv2D: ReLU activation, 32 filters of size 3x3, and input size (35, 35, 3).
- MaxPooling2D: Pool size of 2x2.
- Flatten layer.
- A hidden layer with 256 neurons, ReLU activation.
- Output layer with softmax activation and size 43.

The model also used adam optmizer, a categorical crosentropy loss and the accuracy was measured.

The model exhibited promising accuracy during testing (approximately 0.89). However, that number and a sudden increase between Epoch 1/10 and 2/10 suggested possible overfitting. To address this, a Dropout layer was added, resulting in a dramatic decrease in accuracy (around 0.05), supporting the hypothesis of overfitting.

## Improved Model

To enhance performance, parameters of the layers were adjusted. The Conv2D layer was modified to 128 filters of size 4x4, and the hidden layer was changed to a size of 512. Another Dropout layer was introduced to mitigate overfitting. The revised model configuration is as follows:

- Conv2D: ReLU activation, 128 filters of size 4x4, input size (35, 35, 3).
- Dropout layer with a rate of 0.5.
- MaxPooling2D: Pool size of 2x2.
- Flatten layer.
- A hidden layer with 512 neurons, ReLU activation.
- Another Dropout layer with a rate of 0.5.
- Output layer with softmax activation and size 43.

## Intermediate model

The overall accuracy of the model increased to around 0.82 with the changes applied. This value is great, but looking to improve the performance even futher another convolutional layer was added, that decreased the accuracy to an average of 0.05. That might indicate overfitting or vanishing gradients. The model had the following configuration:

- Conv2D: ReLU activation, 128 filters of size 4x4, input size (35, 35, 3).
- Dropout layer with a rate of .5.
- MaxPooling2D: Pool size of 2x2.
- Conv2D: ReLU activation, 128 filters of size 4x4.
- Flatten layer.
- A hidden layer with 512 neurons, ReLU activation.
- Another Dropout layer with a rate of .5.
- Output layer with softmax activation and size 43.

## Final Model

To solve the problems of the intermediate model and try to improve the model overall accuracy a layer of Batch Normalization was added to the model in such way that our model layers were in the following configuration.

- Conv2D: ReLU activation, 128 filters of size 4x4, input size (35, 35, 3).
- Dropout layer with a rate of .5.
- MaxPooling2D: Pool size of 2x2.
- Batch Normalization layer.
- Conv2D: ReLU activation, 128 filters of size 4x4.
- Flatten layer.
- A hidden layer with 512 neurons, ReLU activation.
- Another Dropout layer with a rate of .5.
- Output layer with softmax activation and size 43.

This adjustment resulted in an accuracy averaging over 0.94, which was deemed an acceptable result for our project.