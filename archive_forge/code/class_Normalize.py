from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Normalize(Operation):

    def __init__(self, axis=-1, order=2):
        super().__init__()
        self.axis = axis
        self.order = order

    def compute_output_spec(self, x):
        return KerasTensor(shape=x.shape)

    def call(self, x):
        return _normalize(x, self.order, self.axis)