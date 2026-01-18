import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _scalar_optimization_initialize(func, xs, args, complex_ok=False):
    """Initialize abscissa, function, and args arrays for elementwise function

    Parameters
    ----------
    func : callable
        An elementwise function with signature

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.
    xs : tuple of arrays
        Finite real abscissa arrays. Must be broadcastable.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.

    Returns
    -------
    xs, fs, args : tuple of arrays
        Broadcasted, writeable, 1D abscissa and function value arrays (or
        NumPy floats, if appropriate). The dtypes of the `xs` and `fs` are
        `xfat`; the dtype of the `args` are unchanged.
    shape : tuple of ints
        Original shape of broadcasted arrays.
    xfat : NumPy dtype
        Result dtype of abscissae, function values, and args determined using
        `np.result_type`, except integer types are promoted to `np.float64`.

    Raises
    ------
    ValueError
        If the result dtype is not that of a real scalar

    Notes
    -----
    Useful for initializing the input of SciPy functions that accept
    an elementwise callable, abscissae, and arguments; e.g.
    `scipy.optimize._chandrupatla`.
    """
    nx = len(xs)
    xas = np.broadcast_arrays(*xs, *args)
    xat = np.result_type(*[xa.dtype for xa in xas])
    xat = np.float64 if np.issubdtype(xat, np.integer) else xat
    xs, args = (xas[:nx], xas[nx:])
    xs = [x.astype(xat, copy=False)[()] for x in xs]
    fs = [np.asarray(func(x, *args)) for x in xs]
    shape = xs[0].shape
    message = 'The shape of the array returned by `func` must be the same as the broadcasted shape of `x` and all other `args`.'
    shapes_equal = [f.shape == shape for f in fs]
    if not np.all(shapes_equal):
        raise ValueError(message)
    xfat = np.result_type(*[f.dtype for f in fs] + [xat])
    if not complex_ok and (not np.issubdtype(xfat, np.floating)):
        raise ValueError('Abscissae and function output must be real numbers.')
    xs = [x.astype(xfat, copy=True)[()] for x in xs]
    fs = [f.astype(xfat, copy=True)[()] for f in fs]
    xs = [x.ravel() for x in xs]
    fs = [f.ravel() for f in fs]
    args = [arg.flatten() for arg in args]
    return (xs, fs, args, shape, xfat)