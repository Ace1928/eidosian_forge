import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _wrap_nan_raise(f):

    def f_raise(x, *args):
        fx = f(x, *args)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = f'The function value at x={x} is NaN; solver cannot continue.'
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx
    f_raise._function_calls = 0
    return f_raise