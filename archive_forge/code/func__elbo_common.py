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
def _elbo_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd):
    iv = 0
    m = vcp_mean[self.ident]
    s = vcp_sd[self.ident]
    iv -= np.sum((vc_mean ** 2 + vc_sd ** 2) * np.exp(2 * (s ** 2 - m))) / 2
    iv -= np.sum(m)
    iv -= 0.5 * (vcp_mean ** 2 + vcp_sd ** 2).sum() / self.vcp_p ** 2
    iv -= 0.5 * (fep_mean ** 2 + fep_sd ** 2).sum() / self.fe_p ** 2
    return iv