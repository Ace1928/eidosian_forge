import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def _quad(func, a, b, args, full_output, epsabs, epsrel, limit, points):
    infbounds = 0
    if b != np.inf and a != -np.inf:
        pass
    elif b == np.inf and a != -np.inf:
        infbounds = 1
        bound = a
    elif b == np.inf and a == -np.inf:
        infbounds = 2
        bound = 0
    elif b != np.inf and a == -np.inf:
        infbounds = -1
        bound = b
    else:
        raise RuntimeError("Infinity comparisons don't work for you.")
    if points is None:
        if infbounds == 0:
            return _quadpack._qagse(func, a, b, args, full_output, epsabs, epsrel, limit)
        else:
            return _quadpack._qagie(func, bound, infbounds, args, full_output, epsabs, epsrel, limit)
    elif infbounds != 0:
        raise ValueError('Infinity inputs cannot be used with break points.')
    else:
        the_points = np.unique(points)
        the_points = the_points[a < the_points]
        the_points = the_points[the_points < b]
        the_points = np.concatenate((the_points, (0.0, 0.0)))
        return _quadpack._qagpe(func, a, b, the_points, args, full_output, epsabs, epsrel, limit)