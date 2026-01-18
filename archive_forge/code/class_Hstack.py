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
class Hstack(Operation):

    def call(self, xs):
        return backend.numpy.hstack(xs)

    def compute_output_spec(self, xs):
        first_shape = xs[0].shape
        total_size_on_axis = 0
        dtypes_to_resolve = []
        for x in xs:
            if not shape_equal(x.shape, first_shape, axis=[1], allow_none=True):
                raise ValueError(f"Every value in `xs` must have the same shape except on the `axis` dim. But found element of shape {x.shape}, which is different from the first element's shape {first_shape}.")
            if total_size_on_axis is None or x.shape[1] is None:
                total_size_on_axis = None
            else:
                total_size_on_axis += x.shape[1]
            dtypes_to_resolve.append(getattr(x, 'dtype', type(x)))
        output_shape = list(first_shape)
        output_shape[1] = total_size_on_axis
        dtype = dtypes.result_type(*dtypes_to_resolve)
        return KerasTensor(output_shape, dtype=dtype)