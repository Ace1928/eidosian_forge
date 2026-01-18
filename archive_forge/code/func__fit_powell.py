from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
def _fit_powell(f, score, start_params, fargs, kwargs, disp=True, maxiter=100, callback=None, retall=False, full_output=True, hess=None):
    """
    Fit using Powell's conjugate direction algorithm.

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    kwargs : dict[str, Any]
        Extra keyword arguments passed to the objective function, i.e.
        objective(x,**kwargs)
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    check_kwargs(kwargs, ('xtol', 'ftol', 'maxfun', 'start_direc'), 'powell')
    xtol = kwargs.setdefault('xtol', 0.0001)
    ftol = kwargs.setdefault('ftol', 0.0001)
    maxfun = kwargs.setdefault('maxfun', None)
    start_direc = kwargs.setdefault('start_direc', None)
    retvals = optimize.fmin_powell(f, start_params, args=fargs, xtol=xtol, ftol=ftol, maxiter=maxiter, maxfun=maxfun, full_output=full_output, disp=disp, retall=retall, callback=callback, direc=start_direc)
    if full_output:
        if not retall:
            xopt, fopt, direc, niter, fcalls, warnflag = retvals
        else:
            xopt, fopt, direc, niter, fcalls, warnflag, allvecs = retvals
        converged = not warnflag
        retvals = {'fopt': fopt, 'direc': direc, 'iterations': niter, 'fcalls': fcalls, 'warnflag': warnflag, 'converged': converged}
        if retall:
            retvals.update({'allvecs': allvecs})
    else:
        xopt = retvals
        retvals = None
    return (xopt, retvals)