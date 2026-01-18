from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class BatchNorm(Operation):

    def __init__(self, axis, epsilon, name=None):
        super().__init__(name)
        self.axis = axis
        self.epsilon = epsilon

    def _check_shape(self, name, shape, expected_shape):
        if shape != expected_shape:
            raise ValueError(f'Arguments `{name}` must be a vector of length `x.shape[axis]`. Expected: `{expected_shape}`. Received: `{shape}.')

    def compute_output_spec(self, x, mean, variance, offset, scale):
        shape = (x.shape[self.axis],)
        self._check_shape('mean', tuple(mean.shape), shape)
        self._check_shape('variance', tuple(variance.shape), shape)
        if offset is not None:
            self._check_shape('offset', tuple(offset.shape), shape)
        if offset is not scale:
            self._check_shape('scale', tuple(scale.shape), shape)
        return KerasTensor(x.shape, dtype=x.dtype)