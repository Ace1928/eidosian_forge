import keras.src as keras
from keras.src.testing_infra import test_utils
class CustomCallModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(1, activation='relu')
        self.dense2 = keras.layers.Dense(1, activation='softmax')

    def call(self, first, second, fiddle_with_output='no', training=True):
        combined = self.dense1(first) + self.dense2(second)
        if fiddle_with_output == 'yes':
            return 10.0 * combined
        else:
            return combined