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
class Diagonal(Operation):

    def __init__(self, offset=0, axis1=0, axis2=1):
        super().__init__()
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def call(self, x):
        return backend.numpy.diagonal(x, offset=self.offset, axis1=self.axis1, axis2=self.axis2)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if len(x_shape) < 2:
            raise ValueError('`diagonal` requires an array of at least two dimensions, but `x` is of shape {x.shape}.')
        shape_2d = [x_shape[self.axis1], x_shape[self.axis2]]
        x_shape[self.axis1] = -1
        x_shape[self.axis2] = -1
        output_shape = list(filter((-1).__ne__, x_shape))
        if None in shape_2d:
            diag_shape = [None]
        else:
            shorter_side = np.minimum(shape_2d[0], shape_2d[1])
            if self.offset > 0:
                remaining = shape_2d[1] - self.offset
            else:
                remaining = shape_2d[0] + self.offset
            diag_shape = [int(np.maximum(0, np.minimum(remaining, shorter_side)))]
        output_shape = output_shape + diag_shape
        return KerasTensor(output_shape, dtype=x.dtype)