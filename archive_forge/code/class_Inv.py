from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Inv(Operation):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return _inv(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        _assert_square(x)
        return KerasTensor(x.shape, x.dtype)