import numpy as np
from ._penalties import NonePenalty
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime
def _handle_scale(self, params, scale=None, **kwds):
    if scale is None:
        if hasattr(self, 'scaletype'):
            mu = self.predict(params)
            scale = self.estimate_scale(mu)
        else:
            scale = 1
    return scale