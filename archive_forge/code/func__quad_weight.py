import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def _quad_weight(func, a, b, args, full_output, epsabs, epsrel, limlst, limit, maxp1, weight, wvar, wopts):
    if weight not in ['cos', 'sin', 'alg', 'alg-loga', 'alg-logb', 'alg-log', 'cauchy']:
        raise ValueError('%s not a recognized weighting function.' % weight)
    strdict = {'cos': 1, 'sin': 2, 'alg': 1, 'alg-loga': 2, 'alg-logb': 3, 'alg-log': 4}
    if weight in ['cos', 'sin']:
        integr = strdict[weight]
        if b != np.inf and a != -np.inf:
            if wopts is None:
                return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output, epsabs, epsrel, limit, maxp1, 1)
            else:
                momcom = wopts[0]
                chebcom = wopts[1]
                return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output, epsabs, epsrel, limit, maxp1, 2, momcom, chebcom)
        elif b == np.inf and a != -np.inf:
            return _quadpack._qawfe(func, a, wvar, integr, args, full_output, epsabs, limlst, limit, maxp1)
        elif b != np.inf and a == -np.inf:
            if weight == 'cos':

                def thefunc(x, *myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return func(*myargs)
            else:

                def thefunc(x, *myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return -func(*myargs)
            args = (func,) + args
            return _quadpack._qawfe(thefunc, -b, wvar, integr, args, full_output, epsabs, limlst, limit, maxp1)
        else:
            raise ValueError('Cannot integrate with this weight from -Inf to +Inf.')
    else:
        if a in [-np.inf, np.inf] or b in [-np.inf, np.inf]:
            message = 'Cannot integrate with this weight over an infinite interval.'
            raise ValueError(message)
        if weight.startswith('alg'):
            integr = strdict[weight]
            return _quadpack._qawse(func, a, b, wvar, integr, args, full_output, epsabs, epsrel, limit)
        else:
            return _quadpack._qawce(func, a, b, wvar, args, full_output, epsabs, epsrel, limit)