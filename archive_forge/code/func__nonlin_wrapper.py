import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
def _nonlin_wrapper(name, jac):
    """
    Construct a solver wrapper with given name and Jacobian approx.

    It inspects the keyword arguments of ``jac.__init__``, and allows to
    use the same arguments in the wrapper function, in addition to the
    keyword arguments of `nonlin_solve`

    """
    signature = _getfullargspec(jac.__init__)
    args, varargs, varkw, defaults, kwonlyargs, kwdefaults, _ = signature
    kwargs = list(zip(args[-len(defaults):], defaults))
    kw_str = ', '.join([f'{k}={v!r}' for k, v in kwargs])
    if kw_str:
        kw_str = ', ' + kw_str
    kwkw_str = ', '.join([f'{k}={k}' for k, v in kwargs])
    if kwkw_str:
        kwkw_str = kwkw_str + ', '
    if kwonlyargs:
        raise ValueError('Unexpected signature %s' % signature)
    wrapper = "\ndef %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None,\n             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,\n             tol_norm=None, line_search='armijo', callback=None, **kw):\n    jac = %(jac)s(%(kwkw)s **kw)\n    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,\n                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,\n                        callback)\n"
    wrapper = wrapper % dict(name=name, kw=kw_str, jac=jac.__name__, kwkw=kwkw_str)
    ns = {}
    ns.update(globals())
    exec(wrapper, ns)
    func = ns[name]
    func.__doc__ = jac.__doc__
    _set_doc(func)
    return func