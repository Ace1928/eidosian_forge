import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
@cache_readonly
def _res_looo(self):
    """collect required results from the LOOO loop

        all results will be attached.
        currently only 'params', 'mse_resid', 'det_cov_params' are stored

        Reestimates the model with endog and exog dropping one observation
        at a time

        This uses a nobs loop, only attributes of the results instance are
        stored.

        Warning: This will need refactoring and API changes to be able to
        add options.
        """
    from statsmodels.sandbox.tools.cross_val import LeaveOneOut
    get_det_cov_params = lambda res: np.linalg.det(res.cov_params())
    endog = self.results.model.endog
    exog = self.results.model.exog
    init_kwds = self.results.model._get_init_kwds()
    freq_weights = init_kwds.pop('freq_weights')
    var_weights = init_kwds.pop('var_weights')
    offset = offset_ = init_kwds.pop('offset')
    exposure = exposure_ = init_kwds.pop('exposure')
    n_trials = init_kwds.pop('n_trials', None)
    if hasattr(init_kwds['family'], 'initialize'):
        is_binomial = True
    else:
        is_binomial = False
    params = np.zeros(exog.shape, dtype=float)
    scale = np.zeros(endog.shape, dtype=float)
    det_cov_params = np.zeros(endog.shape, dtype=float)
    cv_iter = LeaveOneOut(self.nobs)
    for inidx, outidx in cv_iter:
        if offset is not None:
            offset_ = offset[inidx]
        if exposure is not None:
            exposure_ = exposure[inidx]
        if n_trials is not None:
            init_kwds['n_trials'] = n_trials[inidx]
        mod_i = self.model_class(endog[inidx], exog[inidx], offset=offset_, exposure=exposure_, freq_weights=freq_weights[inidx], var_weights=var_weights[inidx], **init_kwds)
        if is_binomial:
            mod_i.family.n = init_kwds['n_trials']
        res_i = mod_i.fit(start_params=self.results.params, method='newton')
        params[outidx] = res_i.params.copy()
        scale[outidx] = res_i.scale
        det_cov_params[outidx] = get_det_cov_params(res_i)
    return dict(params=params, scale=scale, mse_resid=scale, det_cov_params=det_cov_params)