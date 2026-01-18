import collections
import keras.src as keras
class NestedFunctionalInSubclassModel(keras.Model):
    """A functional nested in subclass model."""

    def __init__(self):
        super().__init__(name='nested_functional_in_subclassed_model')
        self.dense1 = keras.layers.Dense(4, activation='relu')
        self.dense2 = keras.layers.Dense(2, activation='relu')
        self.inner_functional_model = get_functional_model()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.inner_functional_model(x)
        return self.dense2(x)