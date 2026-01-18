import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanquantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None, method=None, keepdims=None, *, interpolation=None):
    return (a, q, out)