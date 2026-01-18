from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import (
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import float_like
from . import families
def _fit_irls(self, start_params=None, maxiter=100, tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
    """
        Fits a generalized linear model for a given family using
        iteratively reweighted least squares (IRLS).
        """
    attach_wls = kwargs.pop('attach_wls', False)
    atol = kwargs.get('atol')
    rtol = kwargs.get('rtol', 0.0)
    tol_criterion = kwargs.get('tol_criterion', 'deviance')
    wls_method = kwargs.get('wls_method', 'lstsq')
    atol = tol if atol is None else atol
    endog = self.endog
    wlsexog = self.exog
    if start_params is None:
        start_params = np.zeros(self.exog.shape[1])
        mu = self.family.starting_mu(self.endog)
        lin_pred = self.family.predict(mu)
    else:
        lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
        mu = self.family.fitted(lin_pred)
    self.scale = self.estimate_scale(mu)
    dev = self.family.deviance(self.endog, mu, self.var_weights, self.freq_weights, self.scale)
    if np.isnan(dev):
        raise ValueError('The first guess on the deviance function returned a nan.  This could be a boundary  problem and should be reported.')
    history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
    converged = False
    criterion = history[tol_criterion]
    if maxiter == 0:
        mu = self.family.fitted(lin_pred)
        self.scale = self.estimate_scale(mu)
        wls_results = lm.RegressionResults(self, start_params, None)
        iteration = 0
    for iteration in range(maxiter):
        self.weights = self.iweights * self.n_trials * self.family.weights(mu)
        wlsendog = lin_pred + self.family.link.deriv(mu) * (self.endog - mu) - self._offset_exposure
        wls_mod = reg_tools._MinimalWLS(wlsendog, wlsexog, self.weights, check_endog=True, check_weights=True)
        wls_results = wls_mod.fit(method=wls_method)
        lin_pred = np.dot(self.exog, wls_results.params)
        lin_pred += self._offset_exposure
        mu = self.family.fitted(lin_pred)
        history = self._update_history(wls_results, mu, history)
        self.scale = self.estimate_scale(mu)
        if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
            msg = 'Perfect separation or prediction detected, parameter may not be identified'
            warnings.warn(msg, category=PerfectSeparationWarning)
        converged = _check_convergence(criterion, iteration + 1, atol, rtol)
        if converged:
            break
    self.mu = mu
    if maxiter > 0:
        wls_method2 = 'pinv' if wls_method == 'lstsq' else wls_method
        wls_model = lm.WLS(wlsendog, wlsexog, self.weights)
        wls_results = wls_model.fit(method=wls_method2)
    glm_results = GLMResults(self, wls_results.params, wls_results.normalized_cov_params, self.scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    glm_results.method = 'IRLS'
    glm_results.mle_settings = {}
    glm_results.mle_settings['wls_method'] = wls_method
    glm_results.mle_settings['optimizer'] = glm_results.method
    if maxiter > 0 and attach_wls is True:
        glm_results.results_wls = wls_results
    history['iteration'] = iteration + 1
    glm_results.fit_history = history
    glm_results.converged = converged
    return GLMResultsWrapper(glm_results)