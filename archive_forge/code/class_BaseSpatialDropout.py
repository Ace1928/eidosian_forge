from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.regularization.dropout import Dropout
class BaseSpatialDropout(Dropout):

    def __init__(self, rate, seed=None, name=None, dtype=None):
        super().__init__(rate, seed=seed, name=name, dtype=dtype)

    def call(self, inputs, training=False):
        if training and self.rate > 0:
            return backend.random.dropout(inputs, self.rate, noise_shape=self._get_noise_shape(inputs), seed=self.seed_generator)
        return inputs

    def get_config(self):
        return {'rate': self.rate, 'seed': self.seed, 'name': self.name, 'dtype': self.dtype}