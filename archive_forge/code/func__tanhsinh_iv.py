import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _tanhsinh_iv(f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback):
    message = '`f` must be callable.'
    if not callable(f):
        raise ValueError(message)
    message = 'All elements of `a` and `b` must be real numbers.'
    a, b = np.broadcast_arrays(a, b)
    if np.any(np.iscomplex(a)) or np.any(np.iscomplex(b)):
        raise ValueError(message)
    message = '`log` must be True or False.'
    if log not in {True, False}:
        raise ValueError(message)
    log = bool(log)
    if atol is None:
        atol = -np.inf if log else 0
    rtol_temp = rtol if rtol is not None else 0.0
    params = np.asarray([atol, rtol_temp, 0.0])
    message = '`atol` and `rtol` must be real numbers.'
    if not np.issubdtype(params.dtype, np.floating):
        raise ValueError(message)
    if log:
        message = '`atol` and `rtol` may not be positive infinity.'
        if np.any(np.isposinf(params)):
            raise ValueError(message)
    else:
        message = '`atol` and `rtol` must be non-negative and finite.'
        if np.any(params < 0) or np.any(np.isinf(params)):
            raise ValueError(message)
    atol = params[0]
    rtol = rtol if rtol is None else params[1]
    BIGINT = float(2 ** 62)
    if maxfun is None and maxlevel is None:
        maxlevel = 10
    maxfun = BIGINT if maxfun is None else maxfun
    maxlevel = BIGINT if maxlevel is None else maxlevel
    message = '`maxfun`, `maxlevel`, and `minlevel` must be integers.'
    params = np.asarray([maxfun, maxlevel, minlevel])
    if not (np.issubdtype(params.dtype, np.number) and np.all(np.isreal(params)) and np.all(params.astype(np.int64) == params)):
        raise ValueError(message)
    message = '`maxfun`, `maxlevel`, and `minlevel` must be non-negative.'
    if np.any(params < 0):
        raise ValueError(message)
    maxfun, maxlevel, minlevel = params.astype(np.int64)
    minlevel = min(minlevel, maxlevel)
    if not np.iterable(args):
        args = (args,)
    if callback is not None and (not callable(callback)):
        raise ValueError('`callback` must be callable.')
    return (f, a, b, log, maxfun, maxlevel, minlevel, atol, rtol, args, callback)