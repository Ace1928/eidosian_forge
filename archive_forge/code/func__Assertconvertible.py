import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _Assertconvertible(values, dtype):
    if dtype is None or dtype not in _TF_TO_IS_OK:
        return
    fn_list = _TF_TO_IS_OK.get(dtype)
    mismatch = _FirstNotNone([fn(values) for fn in fn_list])
    if mismatch is not None:
        raise TypeError("Expected %s, got %s of type '%s' instead." % (dtype.name, repr(mismatch), type(mismatch).__name__))