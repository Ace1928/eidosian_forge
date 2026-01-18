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
class _VariationalBayesMixedGLM:
    """
    A mixin providing generic (not family-specific) methods for
    variational Bayes mean field fitting.
    """
    rng = 5
    verbose = False

    def _lp_stats(self, fep_mean, fep_sd, vc_mean, vc_sd):
        tm = np.dot(self.exog, fep_mean)
        tv = np.dot(self.exog ** 2, fep_sd ** 2)
        tm += self.exog_vc.dot(vc_mean)
        tv += self.exog_vc2.dot(vc_sd ** 2)
        return (tm, tv)

    def vb_elbo_base(self, h, tm, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.

        This function calculates the family-specific ELBO function
        based on information provided from a subclass.

        Parameters
        ----------
        h : function mapping 1d vector to 1d vector
            The contribution of the model to the ELBO function can be
            expressed as y_i*lp_i + Eh_i(z), where y_i and lp_i are
            the response and linear predictor for observation i, and z
            is a standard normal random variable.  This formulation
            can be achieved for any GLM with a canonical link
            function.
        """
        iv = 0
        for w in glw:
            z = self.rng * w[1]
            iv += w[0] * h(z) * np.exp(-z ** 2 / 2)
        iv /= np.sqrt(2 * np.pi)
        iv *= self.rng
        iv += self.endog * tm
        iv = iv.sum()
        iv += self._elbo_common(fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd)
        r = iv + np.sum(np.log(fep_sd)) + np.sum(np.log(vcp_sd)) + np.sum(np.log(vc_sd))
        return r

    def vb_elbo_grad_base(self, h, tm, tv, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd):
        """
        Return the gradient of the ELBO function.

        See vb_elbo_base for parameters.
        """
        fep_mean_grad = 0.0
        fep_sd_grad = 0.0
        vcp_mean_grad = 0.0
        vcp_sd_grad = 0.0
        vc_mean_grad = 0.0
        vc_sd_grad = 0.0
        for w in glw:
            z = self.rng * w[1]
            u = h(z) * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
            r = u / np.sqrt(tv)
            fep_mean_grad += w[0] * np.dot(u, self.exog)
            vc_mean_grad += w[0] * self.exog_vc.transpose().dot(u)
            fep_sd_grad += w[0] * z * np.dot(r, self.exog ** 2 * fep_sd)
            v = self.exog_vc2.multiply(vc_sd).transpose().dot(r)
            v = np.squeeze(np.asarray(v))
            vc_sd_grad += w[0] * z * v
        fep_mean_grad *= self.rng
        vc_mean_grad *= self.rng
        fep_sd_grad *= self.rng
        vc_sd_grad *= self.rng
        fep_mean_grad += np.dot(self.endog, self.exog)
        vc_mean_grad += self.exog_vc.transpose().dot(self.endog)
        fep_mean_grad_i, fep_sd_grad_i, vcp_mean_grad_i, vcp_sd_grad_i, vc_mean_grad_i, vc_sd_grad_i = self._elbo_grad_common(fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd)
        fep_mean_grad += fep_mean_grad_i
        fep_sd_grad += fep_sd_grad_i
        vcp_mean_grad += vcp_mean_grad_i
        vcp_sd_grad += vcp_sd_grad_i
        vc_mean_grad += vc_mean_grad_i
        vc_sd_grad += vc_sd_grad_i
        fep_sd_grad += 1 / fep_sd
        vcp_sd_grad += 1 / vcp_sd
        vc_sd_grad += 1 / vc_sd
        mean_grad = np.concatenate((fep_mean_grad, vcp_mean_grad, vc_mean_grad))
        sd_grad = np.concatenate((fep_sd_grad, vcp_sd_grad, vc_sd_grad))
        if self.verbose:
            print('|G|=%f' % np.sqrt(np.sum(mean_grad ** 2) + np.sum(sd_grad ** 2)))
        return (mean_grad, sd_grad)

    def fit_vb(self, mean=None, sd=None, fit_method='BFGS', minim_opts=None, scale_fe=False, verbose=False):
        """
        Fit a model using the variational Bayes mean field approximation.

        Parameters
        ----------
        mean : array_like
            Starting value for VB mean vector
        sd : array_like
            Starting value for VB standard deviation vector
        fit_method : str
            Algorithm for scipy.minimize
        minim_opts : dict
            Options passed to scipy.minimize
        scale_fe : bool
            If true, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.
        verbose : bool
            If True, print the gradient norm to the screen each time
            it is calculated.

        Notes
        -----
        The goal is to find a factored Gaussian approximation
        q1*q2*...  to the posterior distribution, approximately
        minimizing the KL divergence from the factored approximation
        to the actual posterior.  The KL divergence, or ELBO function
        has the form

            E* log p(y, fe, vcp, vc) - E* log q

        where E* is expectation with respect to the product of qj.

        References
        ----------
        Blei, Kucukelbir, McAuliffe (2017).  Variational Inference: A
        review for Statisticians
        https://arxiv.org/pdf/1601.00670.pdf
        """
        self.verbose = verbose
        if scale_fe:
            mn = self.exog.mean(0)
            sc = self.exog.std(0)
            self._exog_save = self.exog
            self.exog = self.exog.copy()
            ixs = np.flatnonzero(sc > 1e-08)
            self.exog[:, ixs] -= mn[ixs]
            self.exog[:, ixs] /= sc[ixs]
        n = self.k_fep + self.k_vcp + self.k_vc
        ml = self.k_fep + self.k_vcp + self.k_vc
        if mean is None:
            m = np.zeros(n)
        else:
            if len(mean) != ml:
                raise ValueError('mean has incorrect length, %d != %d' % (len(mean), ml))
            m = mean.copy()
        if sd is None:
            s = -0.5 + 0.1 * np.random.normal(size=n)
        else:
            if len(sd) != ml:
                raise ValueError('sd has incorrect length, %d != %d' % (len(sd), ml))
            s = np.log(sd)
        i1, i2 = (self.k_fep, self.k_fep + self.k_vcp)
        m[i1:i2] = np.where(m[i1:i2] < -1, -1, m[i1:i2])
        s = np.where(s < -1, -1, s)

        def elbo(x):
            n = len(x) // 2
            return -self.vb_elbo(x[:n], np.exp(x[n:]))

        def elbo_grad(x):
            n = len(x) // 2
            gm, gs = self.vb_elbo_grad(x[:n], np.exp(x[n:]))
            gs *= np.exp(x[n:])
            return -np.concatenate((gm, gs))
        start = np.concatenate((m, s))
        mm = minimize(elbo, start, jac=elbo_grad, method=fit_method, options=minim_opts)
        if not mm.success:
            warnings.warn('VB fitting did not converge')
        n = len(mm.x) // 2
        params = mm.x[0:n]
        va = np.exp(2 * mm.x[n:])
        if scale_fe:
            self.exog = self._exog_save
            del self._exog_save
            params[ixs] /= sc[ixs]
            va[ixs] /= sc[ixs] ** 2
        return BayesMixedGLMResults(self, params, va, mm)

    def _elbo_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd):
        iv = 0
        m = vcp_mean[self.ident]
        s = vcp_sd[self.ident]
        iv -= np.sum((vc_mean ** 2 + vc_sd ** 2) * np.exp(2 * (s ** 2 - m))) / 2
        iv -= np.sum(m)
        iv -= 0.5 * (vcp_mean ** 2 + vcp_sd ** 2).sum() / self.vcp_p ** 2
        iv -= 0.5 * (fep_mean ** 2 + fep_sd ** 2).sum() / self.fe_p ** 2
        return iv

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