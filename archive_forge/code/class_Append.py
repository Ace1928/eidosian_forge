import builtins
import re
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import dtypes
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape
class Append(Operation):

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x1, x2):
        return backend.numpy.append(x1, x2, axis=self.axis)

    def compute_output_spec(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        dtype = dtypes.result_type(getattr(x1, 'dtype', type(x1)), getattr(x2, 'dtype', type(x2)))
        if self.axis is None:
            if None in x1_shape or None in x2_shape:
                output_shape = [None]
            else:
                output_shape = [int(np.prod(x1_shape) + np.prod(x2_shape))]
            return KerasTensor(output_shape, dtype=dtype)
        if not shape_equal(x1_shape, x2_shape, [self.axis]):
            raise ValueError(f'`append` requires inputs to have the same shape except the `axis={self.axis}`, but received shape {x1_shape} and {x2_shape}.')
        output_shape = list(x1_shape)
        output_shape[self.axis] = x1_shape[self.axis] + x2_shape[self.axis]
        return KerasTensor(output_shape, dtype=dtype)