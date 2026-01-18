import keras.src as keras
from keras.src.testing_infra import test_utils
class NestedTestModel2(keras.Model):
    """A model subclass with a functional-API graph network inside."""

    def __init__(self, num_classes=2):
        super().__init__(name='nested_model_2')
        self.num_classes = num_classes
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='relu')
        self.bn = self.bn = keras.layers.BatchNormalization()
        self.test_net = self.get_functional_graph_model(32, 4)

    @staticmethod
    def get_functional_graph_model(input_dim, num_classes):
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(num_classes)(x)
        return keras.Model(inputs, outputs)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn(x)
        x = self.test_net(x)
        return self.dense2(x)