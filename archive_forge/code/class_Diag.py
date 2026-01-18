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
class Diag(Operation):

    def __init__(self, k=0):
        super().__init__()
        self.k = k

    def call(self, x):
        return backend.numpy.diag(x, k=self.k)

    def compute_output_spec(self, x):
        x_shape = x.shape
        if len(x_shape) == 1:
            if x_shape[0] is None:
                output_shape = [None, None]
            else:
                output_shape = [x_shape[0] + int(np.abs(self.k)), x_shape[0] + int(np.abs(self.k))]
        elif len(x_shape) == 2:
            if None in x_shape:
                output_shape = [None]
            else:
                shorter_side = np.minimum(x_shape[0], x_shape[1])
                if self.k > 0:
                    remaining = x_shape[1] - self.k
                else:
                    remaining = x_shape[0] + self.k
                output_shape = [int(np.maximum(0, np.minimum(remaining, shorter_side)))]
        else:
            raise ValueError(f'`x` must be 1-D or 2-D, but received shape {x.shape}.')
        return KerasTensor(output_shape, dtype=x.dtype)