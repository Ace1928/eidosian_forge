import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanmedian_dispatcher(a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)