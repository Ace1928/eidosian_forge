from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance.

        For example, if the the design matrix of a linear model changes then
        initialized can be used to recompute values using the modified design
        matrix.
        """
        pass

    def loglike(self, params):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Notes
        -----
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=100, full_output=True, disp=True, fargs=(), callback=None, retall=False, skip_hessian=False, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver
            - 'minimize' for generic wrapper of scipy minimize (BFGS by default)

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        fargs : tuple, optional
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args)
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool, optional
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        skip_hessian : bool, optional
            If False (default), then the negative inverse hessian is calculated
            after the optimization. If True, then the hessian will not be
            calculated. However, it will be available in methods that use the
            hessian in the optimization (currently only with `"newton"`).
        kwargs : keywords
            All kwargs are passed to the chosen solver with one exception. The
            following keyword controls what happens after the fit::

                warn_convergence : bool, optional
                    If True, checks the model for the converged flag. If the
                    converged flag is False, a ConvergenceWarning is issued.

        Notes
        -----
        The 'basinhopping' solver ignores `maxiter`, `retall`, `full_output`
        explicit arguments.

        Optional arguments for solvers (see returned Results.mle_settings)::

            'newton'
                tol : float
                    Relative error in params acceptable for convergence.
            'nm' -- Nelder Mead
                xtol : float
                    Relative error in params acceptable for convergence
                ftol : float
                    Relative error in loglike(params) acceptable for
                    convergence
                maxfun : int
                    Maximum number of function evaluations to make.
            'bfgs'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.inf is max, -np.inf is min)
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
            'lbfgs'
                m : int
                    This many terms are used for the Hessian approximation.
                factr : float
                    A stop condition that is a variant of relative error.
                pgtol : float
                    A stop condition that uses the projected gradient.
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
                maxfun : int
                    Maximum number of function evaluations to make.
                bounds : sequence
                    (min, max) pairs for each element in x,
                    defining the bounds on that parameter.
                    Use None for one of min or max when there is no bound
                    in that direction.
            'cg'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.inf is max, -np.inf is min)
                epsilon : float
                    If fprime is approximated, use this value for the step
                    size. Can be scalar or vector.  Only relevant if
                    Likelihoodmodel.score is None.
            'ncg'
                fhess_p : callable f'(x,*args)
                    Function which computes the Hessian of f times an arbitrary
                    vector, p.  Should only be supplied if
                    LikelihoodModel.hessian is None.
                avextol : float
                    Stop when the average relative error in the minimizer
                    falls below this amount.
                epsilon : float or ndarray
                    If fhess is approximated, use this value for the step size.
                    Only relevant if Likelihoodmodel.hessian is None.
            'powell'
                xtol : float
                    Line-search error tolerance
                ftol : float
                    Relative error in loglike(params) for acceptable for
                    convergence.
                maxfun : int
                    Maximum number of function evaluations to make.
                start_direc : ndarray
                    Initial direction set.
            'basinhopping'
                niter : int
                    The number of basin hopping iterations.
                niter_success : int
                    Stop the run if the global minimum candidate remains the
                    same for this number of iterations.
                T : float
                    The "temperature" parameter for the accept or reject
                    criterion. Higher "temperatures" mean that larger jumps
                    in function value will be accepted. For best results
                    `T` should be comparable to the separation (in function
                    value) between local minima.
                stepsize : float
                    Initial step size for use in the random displacement.
                interval : int
                    The interval for how often to update the `stepsize`.
                minimizer : dict
                    Extra keyword arguments to be passed to the minimizer
                    `scipy.optimize.minimize()`, for example 'method' - the
                    minimization method (e.g. 'L-BFGS-B'), or 'tol' - the
                    tolerance for termination. Other arguments are mapped from
                    explicit argument of `fit`:
                      - `args` <- `fargs`
                      - `jac` <- `score`
                      - `hess` <- `hess`
            'minimize'
                min_method : str, optional
                    Name of minimization method to use.
                    Any method specific arguments can be passed directly.
                    For a list of methods and their arguments, see
                    documentation of `scipy.optimize.minimize`.
                    If no method is specified, then BFGS is used.
        """
        Hinv = None
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            elif self.exog is not None:
                start_params = [0.0] * self.exog.shape[1]
            else:
                raise ValueError('If exog is None, then start_params should be specified')
        nobs = self.endog.shape[0]

        def f(params, *args):
            return -self.loglike(params, *args) / nobs
        if method == 'newton':

            def score(params, *args):
                return self.score(params, *args) / nobs

            def hess(params, *args):
                return self.hessian(params, *args) / nobs
        else:

            def score(params, *args):
                return -self.score(params, *args) / nobs

            def hess(params, *args):
                return -self.hessian(params, *args) / nobs
        warn_convergence = kwargs.pop('warn_convergence', True)
        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
            if cov_kwds:
                del kwargs['cov_kwds']
            del kwargs['cov_type']
        else:
            kwds = {}
        if 'use_t' in kwargs:
            kwds['use_t'] = kwargs['use_t']
            del kwargs['use_t']
        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score, start_params, fargs, kwargs, hessian=hess, method=method, disp=disp, maxiter=maxiter, callback=callback, retall=retall, full_output=full_output)
        optim_settings.update(kwds)
        cov_params_func = kwargs.setdefault('cov_params_func', None)
        if cov_params_func:
            Hinv = cov_params_func(self, xopt, retvals)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian']) / nobs
        elif not skip_hessian:
            H = -1 * self.hessian(xopt)
            invertible = False
            if np.all(np.isfinite(H)):
                eigvals, eigvecs = np.linalg.eigh(H)
                if np.min(eigvals) > 0:
                    invertible = True
            if invertible:
                Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
                Hinv = np.asfortranarray((Hinv + Hinv.T) / 2.0)
            else:
                warnings.warn('Inverting hessian failed, no bse or cov_params available', HessianInversionWarning)
                Hinv = None
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1.0, **kwds)
        mlefit.mle_retvals = retvals
        if isinstance(retvals, dict):
            if warn_convergence and (not retvals['converged']):
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warnings.warn('Maximum Likelihood optimization failed to converge. Check mle_retvals', ConvergenceWarning)
        mlefit.mle_settings = optim_settings
        return mlefit

    def _fit_zeros(self, keep_index=None, start_params=None, return_auxiliary=False, k_params=None, **fit_kwds):
        """experimental, fit the model subject to zero constraints

        Intended for internal use cases until we know what we need.
        API will need to change to handle models with two exog.
        This is not yet supported by all model subclasses.

        This is essentially a simplified version of `fit_constrained`, and
        does not need to use `offset`.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.

        Some subclasses could use a more efficient calculation than using a
        new model.

        Parameters
        ----------
        keep_index : array_like (int or bool) or slice
            variables that should be dropped.
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        k_params : int or None
            If None, then we try to infer from start_params or model.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """
        if hasattr(self, 'k_extra') and self.k_extra > 0:
            keep_index = np.array(keep_index, copy=True)
            k = self.exog.shape[1]
            extra_index = np.arange(k, k + self.k_extra)
            keep_index_p = np.concatenate((keep_index, extra_index))
        else:
            keep_index_p = keep_index
        if start_params is not None:
            fit_kwds['start_params'] = start_params[keep_index_p]
            k_params = len(start_params)
        init_kwds = self._get_init_kwds()
        mod_constr = self.__class__(self.endog, self.exog[:, keep_index], **init_kwds)
        res_constr = mod_constr.fit(**fit_kwds)
        keep_index = keep_index_p
        if k_params is None:
            k_params = self.exog.shape[1]
            k_params += getattr(self, 'k_extra', 0)
        params_full = np.zeros(k_params)
        params_full[keep_index] = res_constr.params
        try:
            res = self.fit(maxiter=0, disp=0, method='nm', skip_hessian=True, warn_convergence=False, start_params=params_full)
        except (TypeError, ValueError):
            res = self.fit()
        if hasattr(res_constr.model, 'scale'):
            res.model.scale = res._results.scale = res_constr.model.scale
        if hasattr(res_constr, 'mle_retvals'):
            res._results.mle_retvals = res_constr.mle_retvals
        if hasattr(res_constr, 'mle_settings'):
            res._results.mle_settings = res_constr.mle_settings
        res._results.params = params_full
        if not hasattr(res._results, 'normalized_cov_params') or res._results.normalized_cov_params is None:
            res._results.normalized_cov_params = np.zeros((k_params, k_params))
        else:
            res._results.normalized_cov_params[...] = 0
        keep_index = np.array(keep_index)
        res._results.normalized_cov_params[keep_index[:, None], keep_index] = res_constr.normalized_cov_params
        k_constr = res_constr.df_resid - res._results.df_resid
        if hasattr(res_constr, 'cov_params_default'):
            res._results.cov_params_default = np.zeros((k_params, k_params))
            res._results.cov_params_default[keep_index[:, None], keep_index] = res_constr.cov_params_default
        if hasattr(res_constr, 'cov_type'):
            res._results.cov_type = res_constr.cov_type
            res._results.cov_kwds = res_constr.cov_kwds
        res._results.keep_index = keep_index
        res._results.df_resid = res_constr.df_resid
        res._results.df_model = res_constr.df_model
        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr
        if hasattr(res.model, 'M'):
            del res._results._cache['resid']
            del res._results._cache['fittedvalues']
            del res._results._cache['sresid']
            cov = res._results._cache['bcov_scaled']
            cov[...] = 0
            cov[keep_index[:, None], keep_index] = res_constr.bcov_scaled
            res._results.cov_params_default = cov
        return res

    def _fit_collinear(self, atol=1e-14, rtol=1e-13, **kwds):
        """experimental, fit of the model without collinear variables

        This currently uses QR to drop variables based on the given
        sequence.
        Options will be added in future, when the supporting functions
        to identify collinear variables become available.
        """
        x = self.exog
        tol = atol + rtol * x.var(0)
        r = np.linalg.qr(x, mode='r')
        mask = np.abs(r.diagonal()) < np.sqrt(tol)
        idx_keep = np.where(~mask)[0]
        return self._fit_zeros(keep_index=idx_keep, **kwds)