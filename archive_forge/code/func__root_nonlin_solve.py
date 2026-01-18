import numpy as np
from warnings import warn
from ._optimize import MemoizeJac, OptimizeResult, _check_unknown_options
from ._minpack_py import _root_hybr, leastsq
from ._spectral import _root_df_sane
from . import _nonlin as nonlin
def _root_nonlin_solve(fun, x0, args=(), jac=None, _callback=None, _method=None, nit=None, disp=False, maxiter=None, ftol=None, fatol=None, xtol=None, xatol=None, tol_norm=None, line_search='armijo', jac_options=None, **unknown_options):
    _check_unknown_options(unknown_options)
    f_tol = fatol
    f_rtol = ftol
    x_tol = xatol
    x_rtol = xtol
    verbose = disp
    if jac_options is None:
        jac_options = dict()
    jacobian = {'broyden1': nonlin.BroydenFirst, 'broyden2': nonlin.BroydenSecond, 'anderson': nonlin.Anderson, 'linearmixing': nonlin.LinearMixing, 'diagbroyden': nonlin.DiagBroyden, 'excitingmixing': nonlin.ExcitingMixing, 'krylov': nonlin.KrylovJacobian}[_method]
    if args:
        if jac is True:

            def f(x):
                return fun(x, *args)[0]
        else:

            def f(x):
                return fun(x, *args)
    else:
        f = fun
    x, info = nonlin.nonlin_solve(f, x0, jacobian=jacobian(**jac_options), iter=nit, verbose=verbose, maxiter=maxiter, f_tol=f_tol, f_rtol=f_rtol, x_tol=x_tol, x_rtol=x_rtol, tol_norm=tol_norm, line_search=line_search, callback=_callback, full_output=True, raise_exception=False)
    sol = OptimizeResult(x=x, method=_method)
    sol.update(info)
    return sol