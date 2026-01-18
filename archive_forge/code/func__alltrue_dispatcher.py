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
def _alltrue_dispatcher(a, axis=None, out=None, keepdims=None, *, where=None):
    warnings.warn('`alltrue` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `all` instead.', DeprecationWarning, stacklevel=3)
    return (a, where, out)