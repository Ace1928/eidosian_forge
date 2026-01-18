import warnings
import numpy as np
import scipy
from ..exceptions import ConvergenceWarning
from .fixes import line_search_wolfe1, line_search_wolfe2
def _newton_cg(grad_hess, func, grad, x0, args=(), tol=0.0001, maxiter=100, maxinner=200, line_search=True, warn=True):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Parameters
    ----------
    grad_hess : callable
        Should return the gradient and a callable returning the matvec product
        of the Hessian.

    func : callable
        Should return the value of the function.

    grad : callable
        Should return the function value and the gradient. This is used
        by the linesearch functions.

    x0 : array of float
        Initial guess.

    args : tuple, default=()
        Arguments passed to func_grad_hess, func and grad.

    tol : float, default=1e-4
        Stopping criterion. The iteration will stop when
        ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    maxiter : int, default=100
        Number of Newton iterations.

    maxinner : int, default=200
        Number of CG iterations.

    line_search : bool, default=True
        Whether to use a line search or not.

    warn : bool, default=True
        Whether to warn when didn't converge.

    Returns
    -------
    xk : ndarray of float
        Estimated minimum.
    """
    x0 = np.asarray(x0).flatten()
    xk = np.copy(x0)
    k = 0
    if line_search:
        old_fval = func(x0, *args)
        old_old_fval = None
    while k < maxiter:
        fgrad, fhess_p = grad_hess(xk, *args)
        absgrad = np.abs(fgrad)
        if np.max(absgrad) <= tol:
            break
        maggrad = np.sum(absgrad)
        eta = min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = _cg(fhess_p, fgrad, maxiter=maxinner, tol=termcond)
        alphak = 1.0
        if line_search:
            try:
                alphak, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(func, grad, xk, xsupi, fgrad, old_fval, old_old_fval, args=args)
            except _LineSearchError:
                warnings.warn('Line Search failed')
                break
        xk += alphak * xsupi
        k += 1
    if warn and k >= maxiter:
        warnings.warn('newton-cg failed to converge. Increase the number of iterations.', ConvergenceWarning)
    return (xk, k)