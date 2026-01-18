from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
def _assert_a_b_compat(a, b):
    if a.ndim == b.ndim:
        if a.shape[-2] != b.shape[-2]:
            raise ValueError(f'Incompatible shapes between `a` and `b`. Expected `a.shape[-2] == b.shape[-2]`. Received: a.shape={a.shape}, b.shape={b.shape}')
    elif a.ndim == b.ndim - 1:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(f'Incompatible shapes between `a` and `b`. Expected `a.shape[-1] == b.shape[-1]`. Received: a.shape={a.shape}, b.shape={b.shape}')