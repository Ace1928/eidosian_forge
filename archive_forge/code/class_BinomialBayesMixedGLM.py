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
class BinomialBayesMixedGLM(_VariationalBayesMixedGLM, _BayesMixedGLM):
    __doc__ = _init_doc.format(example=_logit_example)

    def __init__(self, endog, exog, exog_vc, ident, vcp_p=1, fe_p=2, fep_names=None, vcp_names=None, vc_names=None):
        super().__init__(endog, exog, exog_vc=exog_vc, ident=ident, vcp_p=vcp_p, fe_p=fe_p, family=families.Binomial(), fep_names=fep_names, vcp_names=vcp_names, vc_names=vc_names)
        if not np.all(np.unique(endog) == np.r_[0, 1]):
            msg = 'endog values must be 0 and 1, and not all identical'
            raise ValueError(msg)

    @classmethod
    def from_formula(cls, formula, vc_formulas, data, vcp_p=1, fe_p=2):
        fam = families.Binomial()
        x = _BayesMixedGLM.from_formula(formula, vc_formulas, data, family=fam, vcp_p=vcp_p, fe_p=fe_p)
        mod = BinomialBayesMixedGLM(x.endog, x.exog, exog_vc=x.exog_vc, ident=x.ident, vcp_p=x.vcp_p, fe_p=x.fe_p, fep_names=x.fep_names, vcp_names=x.vcp_names, vc_names=x.vc_names)
        mod.data = x.data
        return mod

    def vb_elbo(self, vb_mean, vb_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.
        """
        fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
        fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)
        tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

        def h(z):
            return -np.log(1 + np.exp(tm + np.sqrt(tv) * z))
        return self.vb_elbo_base(h, tm, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd)

    def vb_elbo_grad(self, vb_mean, vb_sd):
        """
        Returns the gradient of the model's evidence lower bound (ELBO).
        """
        fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
        fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)
        tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

        def h(z):
            u = tm + np.sqrt(tv) * z
            x = np.zeros_like(u)
            ii = np.flatnonzero(u > 0)
            uu = u[ii]
            x[ii] = 1 / (1 + np.exp(-uu))
            ii = np.flatnonzero(u <= 0)
            uu = u[ii]
            x[ii] = np.exp(uu) / (1 + np.exp(uu))
            return -x
        return self.vb_elbo_grad_base(h, tm, tv, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd)