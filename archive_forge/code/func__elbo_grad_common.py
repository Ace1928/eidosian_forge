from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def _elbo_grad_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd):
    m = vcp_mean[self.ident]
    s = vcp_sd[self.ident]
    u = vc_mean ** 2 + vc_sd ** 2
    ve = np.exp(2 * (s ** 2 - m))
    dm = u * ve - 1
    ds = -2 * u * ve * s
    vcp_mean_grad = np.bincount(self.ident, weights=dm)
    vcp_sd_grad = np.bincount(self.ident, weights=ds)
    vc_mean_grad = -vc_mean.copy() * ve
    vc_sd_grad = -vc_sd.copy() * ve
    vcp_mean_grad -= vcp_mean / self.vcp_p ** 2
    vcp_sd_grad -= vcp_sd / self.vcp_p ** 2
    fep_mean_grad = -fep_mean.copy() / self.fe_p ** 2
    fep_sd_grad = -fep_sd.copy() / self.fe_p ** 2
    return (fep_mean_grad, fep_sd_grad, vcp_mean_grad, vcp_sd_grad, vc_mean_grad, vc_sd_grad)