from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class Moments(Operation):

    def __init__(self, axes, keepdims=False, synchronized=False, name=None):
        super().__init__(name)
        self.axes = axes
        self.keepdims = keepdims
        self.synchronized = synchronized

    def call(self, x):
        return backend.nn.moments(x, axes=self.axes, keepdims=self.keepdims, synchronized=self.synchronized)

    def compute_output_spec(self, x):
        return (KerasTensor(reduce_shape(x.shape, axis=self.axes, keepdims=self.keepdims), dtype=x.dtype), KerasTensor(reduce_shape(x.shape, axis=self.axes, keepdims=self.keepdims), dtype=x.dtype))