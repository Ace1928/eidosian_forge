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
class GetItem(Operation):

    def call(self, x, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[key]

    def compute_output_spec(self, x, key):
        remaining_shape = list(x.shape)
        new_shape = []
        if isinstance(key, int):
            remaining_key = [key]
        elif isinstance(key, tuple):
            remaining_key = list(key)
        else:
            raise ValueError(f'Unsupported key type for array slice. Recieved: `{key}`')
        num_ellipses = remaining_key.count(Ellipsis)
        if num_ellipses > 1:
            raise ValueError(f'Slice should only have one ellipsis. Recieved: `{key}`')
        elif num_ellipses == 0:
            remaining_key.append(Ellipsis)
        while True:
            if not remaining_key:
                break
            subkey = remaining_key.pop(0)
            if subkey == Ellipsis:
                needed = len(remaining_key) - remaining_key.count(np.newaxis)
                consumed = len(remaining_shape) - needed
                new_shape += remaining_shape[:consumed]
                remaining_shape = remaining_shape[consumed:]
                continue
            if subkey == np.newaxis:
                new_shape.append(1)
                continue
            if not remaining_shape:
                raise ValueError(f'Array has shape {x.shape} but slice has to many indices. Recieved: `{key}`')
            length = remaining_shape.pop(0)
            if isinstance(subkey, int):
                if length is not None:
                    index = subkey if subkey >= 0 else subkey + length
                    if index < 0 or index >= length:
                        raise ValueError(f'Array has shape {x.shape} but out-of-bounds index {key} was requested.')
            elif isinstance(subkey, slice):
                if length is not None:
                    new_length = len(range(*subkey.indices(length)))
                    new_shape.append(new_length)
                else:
                    new_shape.append(length)
            else:
                raise ValueError(f'Unsupported key type for array slice. Recieved: `{key}`')
        return KerasTensor(tuple(new_shape), dtype=x.dtype)