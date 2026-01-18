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
class Arange(Operation):

    def call(self, start, stop=None, step=1, dtype=None):
        return backend.numpy.arange(start, stop, step=step, dtype=dtype)

    def compute_output_spec(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            start, stop = (0, start)
        output_shape = [int(np.ceil((stop - start) / step))]
        if dtype is None:
            dtypes_to_resolve = [getattr(start, 'dtype', type(start)), getattr(step, 'dtype', type(step))]
            if stop is not None:
                dtypes_to_resolve.append(getattr(stop, 'dtype', type(stop)))
            dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)