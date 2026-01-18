from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Erf(Operation):

    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape, dtype=x.dtype)

    def call(self, x):
        return backend.math.erf(x)