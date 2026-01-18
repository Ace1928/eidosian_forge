from typing import Any, Callable, Tuple
import numpy
from thinc.backends import Ops
from ..config import registry
from ..model import Model
def _to_device(ops, X):
    if isinstance(X, tuple):
        return tuple([_to_device(ops, x) for x in X])
    elif isinstance(X, list):
        return [_to_device(ops, x) for x in X]
    else:
        return ops.asarray(X)