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
def _any_dispatcher(a, axis=None, out=None, keepdims=None, *, where=np._NoValue):
    return (a, where, out)