import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _FilterComplex(v):
    if isinstance(v, (list, tuple)):
        return _FirstNotNone([_FilterComplex(x) for x in v])
    return None if isinstance(v, compat.complex_types) else _NotNone(v)