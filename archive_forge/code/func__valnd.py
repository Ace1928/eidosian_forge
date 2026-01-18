import operator
import functools
import warnings
import numpy as np
from numpy.core.multiarray import dragon4_positional, dragon4_scientific
from numpy.core.umath import absolute
def _valnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>val<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>val<n>d`` functions for more detail
    """
    args = [np.asanyarray(a) for a in args]
    shape0 = args[0].shape
    if not all((a.shape == shape0 for a in args[1:])):
        if len(args) == 3:
            raise ValueError('x, y, z are incompatible')
        elif len(args) == 2:
            raise ValueError('x, y are incompatible')
        else:
            raise ValueError('ordinates are incompatible')
    it = iter(args)
    x0 = next(it)
    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c