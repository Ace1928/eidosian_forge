import functools
import types
import warnings
import numpy as np
from .._utils import set_module
from . import multiarray as mu
from . import overrides
from . import umath as um
from . import numerictypes as nt
from .multiarray import asarray, array, asanyarray, concatenate
from . import _methods
def _sometrue_dispatcher(a, axis=None, out=None, keepdims=None, *, where=np._NoValue):
    warnings.warn('`sometrue` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `any` instead.', DeprecationWarning, stacklevel=3)
    return (a, where, out)