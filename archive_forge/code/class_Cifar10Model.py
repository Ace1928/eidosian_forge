from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ray import train, tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining
class Cifar10Model(Trainable):

    def _read_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.astype('float32')
        x_test /= 255
        return ((x_train, y_train), (x_test, y_test))

    def _build_model(self, input_shape):
        x = Input(shape=(32, 32, 3))
        y = x
        y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
        y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
        y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
        y = Flatten()(y)
        y = Dropout(self.config.get('dropout', 0.5))(y)
        y = Dense(units=10, activation='softmax', kernel_initializer='he_normal')(y)
        model = Model(inputs=x, outputs=y, name='model1')
        return model

    def setup(self, config):
        self.train_data, self.test_data = self._read_data()
        x_train = self.train_data[0]
        model = self._build_model(x_train.shape[1:])
        opt = tf.keras.optimizers.Adadelta(lr=self.config.get('lr', 0.0001), weight_decay=self.config.get('decay', 0.0001))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model = model

    def step(self):
        x_train, y_train = self.train_data
        x_train, y_train = (x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES])
        x_test, y_test = self.test_data
        x_test, y_test = (x_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES])
        aug_gen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False)
        aug_gen.fit(x_train)
        batch_size = self.config.get('batch_size', 64)
        gen = aug_gen.flow(x_train, y_train, batch_size=batch_size)
        self.model.fit_generator(generator=gen, epochs=self.config.get('epochs', 1), validation_data=None)
        _, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {'mean_accuracy': accuracy}

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + '/model'
        self.model.save(file_path)

    def load_checkpoint(self, checkpoint_dir):
        del self.model
        file_path = checkpoint_dir + '/model'
        self.model = load_model(file_path)

    def cleanup(self):
        pass