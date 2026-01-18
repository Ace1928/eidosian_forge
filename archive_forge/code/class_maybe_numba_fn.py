import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
class maybe_numba_fn:
    """Wrap a function to (maybe) use a (lazy) jit-compiled version."""

    def __init__(self, function, **kwargs):
        """Wrap a function and save compilation keywords."""
        self.function = function
        kwargs.setdefault('nopython', True)
        self.kwargs = kwargs

    @lazy_property
    def numba_fn(self):
        """Memoized compiled function."""
        try:
            numba = importlib.import_module('numba')
            numba_fn = numba.jit(**self.kwargs)(self.function)
        except ImportError:
            numba_fn = self.function
        return numba_fn

    def __call__(self, *args, **kwargs):
        """Call the jitted function or normal, depending on flag."""
        if Numba.numba_flag:
            return self.numba_fn(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)