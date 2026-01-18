from typing import Any, Callable, Tuple
import numpy
from thinc.backends import Ops
from ..config import registry
from ..model import Model
def _to_cpu(X):
    if isinstance(X, numpy.ndarray):
        return X
    elif isinstance(X, tuple):
        return tuple([_to_cpu(x) for x in X])
    elif isinstance(X, list):
        return [_to_cpu(x) for x in X]
    elif hasattr(X, 'get'):
        return X.get()
    else:
        return X