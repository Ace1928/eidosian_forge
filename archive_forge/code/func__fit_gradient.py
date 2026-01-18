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
def _fit_gradient(self, start_params=None, method='newton', maxiter=100, tol=1e-08, full_output=True, disp=True, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, max_start_irls=3, **kwargs):
    """
        Fits a generalized linear model for a given family iteratively
        using the scipy gradient optimizers.
        """
    scaletype = self.scaletype
    self.scaletype = 1.0
    if max_start_irls > 0 and start_params is None:
        irls_rslt = self._fit_irls(start_params=start_params, maxiter=max_start_irls, tol=tol, scale=1.0, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs)
        start_params = irls_rslt.params
        del irls_rslt
    rslt = super().fit(start_params=start_params, maxiter=maxiter, full_output=full_output, method=method, disp=disp, **kwargs)
    self.scaletype = scaletype
    mu = self.predict(rslt.params)
    scale = self.estimate_scale(mu)
    if rslt.normalized_cov_params is None:
        cov_p = None
    else:
        cov_p = rslt.normalized_cov_params / scale
    if cov_type.lower() == 'eim':
        oim = False
        cov_type = 'nonrobust'
    else:
        oim = True
    try:
        cov_p = np.linalg.inv(-self.hessian(rslt.params, observed=oim)) / scale
    except LinAlgError:
        warnings.warn('Inverting hessian failed, no bse or cov_params available', HessianInversionWarning)
        cov_p = None
    results_class = getattr(self, '_results_class', GLMResults)
    results_class_wrapper = getattr(self, '_results_class_wrapper', GLMResultsWrapper)
    glm_results = results_class(self, rslt.params, cov_p, scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    history = {'iteration': 0}
    if full_output:
        glm_results.mle_retvals = rslt.mle_retvals
        if 'iterations' in rslt.mle_retvals:
            history['iteration'] = rslt.mle_retvals['iterations']
    glm_results.method = method
    glm_results.fit_history = history
    return results_class_wrapper(glm_results)