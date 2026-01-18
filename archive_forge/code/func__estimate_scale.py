import numpy as np
import scipy.stats as stats
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _estimate_scale(self, resid):
    """
        Estimates the scale based on the option provided to the fit method.
        """
    if isinstance(self.scale_est, str):
        if self.scale_est.lower() == 'mad':
            return scale.mad(resid, center=0)
        else:
            raise ValueError('Option %s for scale_est not understood' % self.scale_est)
    elif isinstance(self.scale_est, scale.HuberScale):
        return self.scale_est(self.df_resid, self.nobs, resid)
    else:
        return scale.scale_est(self, resid) ** 2