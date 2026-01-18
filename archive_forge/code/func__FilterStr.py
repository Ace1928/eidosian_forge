import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _FilterStr(v):
    if isinstance(v, (list, tuple)):
        return _FirstNotNone([_FilterStr(x) for x in v])
    if isinstance(v, compat.bytes_or_text_types):
        return None
    else:
        return _NotNone(v)