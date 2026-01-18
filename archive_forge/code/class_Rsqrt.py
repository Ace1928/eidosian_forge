from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Rsqrt(Operation):

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return backend.math.rsqrt(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype=x.dtype)