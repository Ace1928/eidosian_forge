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
class TakeAlongAxis(Operation):

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x, indices):
        return backend.numpy.take_along_axis(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        x_shape = list(x.shape)
        indices_shape = list(indices.shape)
        if self.axis is None:
            x_shape = [None] if None in x_shape else [int(np.prod(x_shape))]
        if len(x_shape) != len(indices_shape):
            raise ValueError(f'`x` and `indices` must have the same number of dimensions, but receive shape {x_shape} and {indices_shape}.')
        del x_shape[self.axis]
        del indices_shape[self.axis]
        output_shape = broadcast_shapes(x_shape, indices_shape)
        size_on_axis = indices.shape[self.axis]
        if self.axis == -1:
            output_shape = output_shape + [size_on_axis]
        elif self.axis >= 0:
            output_shape.insert(self.axis, size_on_axis)
        else:
            output_shape.insert(self.axis + 1, size_on_axis)
        return KerasTensor(output_shape, dtype=x.dtype)