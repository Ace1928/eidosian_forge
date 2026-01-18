from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
class Mish(ops.Operation):

    def call(self, x):
        return self.static_call(x)

    def compute_output_spec(self, x):
        return backend.KerasTensor(x.shape, x.dtype)

    @staticmethod
    def static_call(x):
        return x * backend.nn.tanh(backend.nn.softplus(x))