from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
class GLMGam(PenalizedMixin, GLM):
    """
    Generalized Additive Models (GAM)

    This inherits from `GLM`.

    Warning: Not all inherited methods might take correctly account of the
    penalization. Not all options including offset and exposure have been
    verified yet.

    Parameters
    ----------
    endog : array_like
        The response variable.
    exog : array_like or None
        This explanatory variables are treated as linear. The model in this
        case is a partial linear model.
    smoother : instance of additive smoother class
        Examples of smoother instances include Bsplines or CyclicCubicSplines.
    alpha : float or list of floats
        Penalization weights for smooth terms. The length of the list needs
        to be the same as the number of smooth terms in the ``smoother``.
    family : instance of GLM family
        See GLM.
    offset : None or array_like
        See GLM.
    exposure : None or array_like
        See GLM.
    missing : 'none'
        Missing value handling is not supported in this class.
    **kwargs
        Extra keywords are used in call to the super classes.

    Notes
    -----
    Status: experimental. This has full unit test coverage for the core
    results with Gaussian and Poisson (without offset and exposure). Other
    options and additional results might not be correctly supported yet.
    (Binomial with counts, i.e. with n_trials, is most likely wrong in pirls.
    User specified var or freq weights are most likely also not correct for
    all results.)
    """
    _results_class = GLMGamResults
    _results_class_wrapper = GLMGamResultsWrapper

    def __init__(self, endog, exog=None, smoother=None, alpha=0, family=None, offset=None, exposure=None, missing='none', **kwargs):
        hasconst = kwargs.get('hasconst', None)
        xnames_linear = None
        if hasattr(exog, 'design_info'):
            self.design_info_linear = exog.design_info
            xnames_linear = self.design_info_linear.column_names
        is_pandas = _is_using_pandas(exog, None)
        self.data_linear = self._handle_data(endog, exog, missing, hasconst)
        if xnames_linear is None:
            xnames_linear = self.data_linear.xnames
        if exog is not None:
            exog_linear = self.data_linear.exog
            k_exog_linear = exog_linear.shape[1]
        else:
            exog_linear = None
            k_exog_linear = 0
        self.k_exog_linear = k_exog_linear
        self.exog_linear = exog_linear
        self.smoother = smoother
        self.k_smooths = smoother.k_variables
        self.alpha = self._check_alpha(alpha)
        penal = MultivariateGamPenalty(smoother, alpha=self.alpha, start_idx=k_exog_linear)
        kwargs.pop('penal', None)
        if exog_linear is not None:
            exog = np.column_stack((exog_linear, smoother.basis))
        else:
            exog = smoother.basis
        if xnames_linear is None:
            xnames_linear = []
        xnames = xnames_linear + self.smoother.col_names
        if is_pandas and exog_linear is not None:
            exog = pd.DataFrame(exog, index=self.data_linear.row_labels, columns=xnames)
        super().__init__(endog, exog=exog, family=family, offset=offset, exposure=exposure, penal=penal, missing=missing, **kwargs)
        if not is_pandas:
            self.exog_names[:] = xnames
        if hasattr(self.data, 'design_info'):
            del self.data.design_info
        if hasattr(self, 'formula'):
            self.formula_linear = self.formula
            self.formula = None
            del self.formula

    def _check_alpha(self, alpha):
        """check and convert alpha to required list format

        Parameters
        ----------
        alpha : scalar, list or array_like
            penalization weight

        Returns
        -------
        alpha : list
            penalization weight, list with length equal to the number of
            smooth terms
        """
        if not isinstance(alpha, Iterable):
            alpha = [alpha] * len(self.smoother.smoothers)
        elif not isinstance(alpha, list):
            alpha = list(alpha)
        return alpha

    def fit(self, start_params=None, maxiter=1000, method='pirls', tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, full_output=True, disp=False, max_start_irls=3, **kwargs):
        """estimate parameters and create instance of GLMGamResults class

        Parameters
        ----------
        most parameters are the same as for GLM
        method : optimization method
            The special optimization method is "pirls" which uses a penalized
            version of IRLS. Other methods are gradient optimizers as used in
            base.model.LikelihoodModel.

        Returns
        -------
        res : instance of wrapped GLMGamResults
        """
        if hasattr(self, 'formula'):
            self.formula_linear = self.formula
            del self.formula
        if method.lower() in ['pirls', 'irls']:
            res = self._fit_pirls(self.alpha, start_params=start_params, maxiter=maxiter, tol=tol, scale=scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t, **kwargs)
        else:
            if max_start_irls > 0 and start_params is None:
                res = self._fit_pirls(self.alpha, start_params=start_params, maxiter=max_start_irls, tol=tol, scale=scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t, **kwargs)
                start_params = res.params
                del res
            res = super().fit(start_params=start_params, maxiter=maxiter, method=method, tol=tol, scale=scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t, full_output=full_output, disp=disp, max_start_irls=0, **kwargs)
        return res

    def _fit_pirls(self, alpha, start_params=None, maxiter=100, tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):
        """fit model with penalized reweighted least squares
        """
        endog = self.endog
        wlsexog = self.exog
        spl_s = self.penal.penalty_matrix(alpha=alpha)
        nobs, n_columns = wlsexog.shape
        if weights is None:
            self.data_weights = np.array([1.0] * nobs)
        else:
            self.data_weights = weights
        if not hasattr(self, '_offset_exposure'):
            self._offset_exposure = 0
        self.scaletype = scale
        self.scale = 1
        if start_params is None:
            mu = self.family.starting_mu(endog)
            lin_pred = self.family.predict(mu)
        else:
            lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
        dev = self.family.deviance(endog, mu)
        history = dict(params=[None, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history['deviance']
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            iteration = 0
        for iteration in range(maxiter):
            self.weights = self.data_weights * self.family.weights(mu)
            wlsendog = lin_pred + self.family.link.deriv(mu) * (endog - mu) - self._offset_exposure
            wls_results = penalized_wls(wlsendog, wlsexog, spl_s, self.weights)
            lin_pred = np.dot(wlsexog, wls_results.params).ravel()
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)
            history = self._update_history(wls_results, mu, history)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = 'Perfect separation detected, results not available'
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration, tol, 0)
            if converged:
                break
        self.mu = mu
        self.scale = self.estimate_scale(mu)
        glm_results = GLMGamResults(self, wls_results.params, wls_results.normalized_cov_params, self.scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        glm_results.method = 'PIRLS'
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged
        return GLMGamResultsWrapper(glm_results)

    def select_penweight(self, criterion='aic', start_params=None, start_model_params=None, method='basinhopping', **fit_kwds):
        """find alpha by minimizing results criterion

        The objective for the minimization can be results attributes like
        ``gcv``, ``aic`` or ``bic`` where the latter are based on effective
        degrees of freedom.

        Warning: In many case the optimization might converge to a local
        optimum or near optimum. Different start_params or using a global
        optimizer is recommended, default is basinhopping.

        Parameters
        ----------
        criterion='aic'
            name of results attribute to be minimized.
            Default is 'aic', other options are 'gcv', 'cv' or 'bic'.
        start_params : None or array
            starting parameters for alpha in the penalization weight
            minimization. The parameters are internally exponentiated and
            the minimization is with respect to ``exp(alpha)``
        start_model_params : None or array
            starting parameter for the ``model._fit_pirls``.
        method : 'basinhopping', 'nm' or 'minimize'
            'basinhopping' and 'nm' directly use the underlying scipy.optimize
            functions `basinhopping` and `fmin`. 'minimize' provides access
            to the high level interface, `scipy.optimize.minimize`.
        fit_kwds : keyword arguments
            additional keyword arguments will be used in the call to the
            scipy optimizer. Which keywords are supported depends on the
            scipy optimization function.

        Returns
        -------
        alpha : ndarray
            penalization parameter found by minimizing the criterion.
            Note that this can be only a local (near) optimum.
        fit_res : tuple
            results returned by the scipy optimization routine. The
            parameters in the optimization problem are `log(alpha)`
        history : dict
            history of calls to pirls and contains alpha, the fit
            criterion and the parameters to which pirls converged to for the
            given alpha.

        Notes
        -----
        In the test cases Nelder-Mead and bfgs often converge to local optima,
        see also https://github.com/statsmodels/statsmodels/issues/5381.

        This does not use any analytical derivatives for the criterion
        minimization.

        Status: experimental, It is possible that defaults change if there
        is a better way to find a global optimum. API (e.g. type of return)
        might also change.
        """
        scale_keep = self.scale
        scaletype_keep = self.scaletype
        alpha_keep = copy.copy(self.alpha)
        if start_params is None:
            start_params = np.zeros(self.k_smooths)
        else:
            start_params = np.log(1e-20 + start_params)
        history = {}
        history['alpha'] = []
        history['params'] = [start_model_params]
        history['criterion'] = []

        def fun(p):
            a = np.exp(p)
            res_ = self._fit_pirls(start_params=history['params'][-1], alpha=a)
            history['alpha'].append(a)
            history['params'].append(np.asarray(res_.params))
            return getattr(res_, criterion)
        if method == 'nm':
            kwds = dict(full_output=True, maxiter=1000, maxfun=2000)
            kwds.update(fit_kwds)
            fit_res = optimize.fmin(fun, start_params, **kwds)
            opt = fit_res[0]
        elif method == 'basinhopping':
            kwds = dict(minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 100, 'maxfev': 500}}, niter=10)
            kwds.update(fit_kwds)
            fit_res = optimize.basinhopping(fun, start_params, **kwds)
            opt = fit_res.x
        elif method == 'minimize':
            fit_res = optimize.minimize(fun, start_params, **fit_kwds)
            opt = fit_res.x
        else:
            raise ValueError('method not recognized')
        del history['params'][0]
        alpha = np.exp(opt)
        self.scale = scale_keep
        self.scaletype = scaletype_keep
        self.alpha = alpha_keep
        return (alpha, fit_res, history)

    def select_penweight_kfold(self, alphas=None, cv_iterator=None, cost=None, k_folds=5, k_grid=11):
        """find alphas by k-fold cross-validation

        Warning: This estimates ``k_folds`` models for each point in the
            grid of alphas.

        Parameters
        ----------
        alphas : None or list of arrays
        cv_iterator : instance
            instance of a cross-validation iterator, by default this is a
            KFold instance
        cost : function
            default is mean squared error. The cost function to evaluate the
            prediction error for the left out sample. This should take two
            arrays as argument and return one float.
        k_folds : int
            number of folds if default Kfold iterator is used.
            This is ignored if ``cv_iterator`` is not None.

        Returns
        -------
        alpha_cv : list of float
            Best alpha in grid according to cross-validation
        res_cv : instance of MultivariateGAMCVPath
            The instance was used for cross-validation and holds the results

        Notes
        -----
        The default alphas are defined as
        ``alphas = [np.logspace(0, 7, k_grid) for _ in range(k_smooths)]``
        """
        if cost is None:

            def cost(x1, x2):
                return np.linalg.norm(x1 - x2) / len(x1)
        if alphas is None:
            alphas = [np.logspace(0, 7, k_grid) for _ in range(self.k_smooths)]
        if cv_iterator is None:
            cv_iterator = KFold(k_folds=k_folds, shuffle=True)
        gam_cv = MultivariateGAMCVPath(smoother=self.smoother, alphas=alphas, gam=GLMGam, cost=cost, endog=self.endog, exog=self.exog_linear, cv_iterator=cv_iterator)
        gam_cv_res = gam_cv.fit()
        return (gam_cv_res.alpha_cv, gam_cv_res)