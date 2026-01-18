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
class Tri(Operation):

    def call(self, N, M=None, k=0, dtype=None):
        return backend.numpy.tri(N, M=M, k=k, dtype=dtype)

    def compute_output_spec(self, N, M=None, k=0, dtype=None):
        if M is None:
            M = N
        dtype = dtype or backend.floatx()
        return KerasTensor((N, M), dtype=dtype)