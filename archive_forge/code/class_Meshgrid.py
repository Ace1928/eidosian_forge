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
class Meshgrid(Operation):

    def __init__(self, indexing='xy'):
        super().__init__()
        if indexing not in ('xy', 'ij'):
            raise ValueError("Valid values for `indexing` are 'xy' and 'ij', but received {index}.")
        self.indexing = indexing

    def call(self, *x):
        return backend.numpy.meshgrid(*x, indexing=self.indexing)

    def compute_output_spec(self, *x):
        output_shape = []
        for xi in x:
            if len(xi.shape) == 0:
                size = 1
            elif None in xi.shape:
                size = None
            else:
                size = int(np.prod(xi.shape))
            output_shape.append(size)
        if self.indexing == 'ij':
            return [KerasTensor(output_shape) for _ in range(len(x))]
        tmp = output_shape[0]
        output_shape[0] = output_shape[1]
        output_shape[1] = tmp
        return [KerasTensor(output_shape, dtype=xi.dtype) for _ in range(len(x))]