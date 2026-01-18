import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _callf(self, x, error=True):
    """Call the user-supplied function, update book-keeping"""
    fx = self.f(x, *self.args)
    self.function_calls += 1
    if not np.isfinite(fx) and error:
        raise ValueError(f'Invalid function value: f({x:f}) -> {fx} ')
    return fx