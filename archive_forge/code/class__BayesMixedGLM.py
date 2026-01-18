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
class _BayesMixedGLM(base.Model):

    def __init__(self, endog, exog, exog_vc=None, ident=None, family=None, vcp_p=1, fe_p=2, fep_names=None, vcp_names=None, vc_names=None, **kwargs):
        if exog.ndim == 1:
            if isinstance(exog, np.ndarray):
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)
        if exog.ndim != 2:
            msg = "'exog' must have one or two columns"
            raise ValueError(msg)
        if exog_vc.ndim == 1:
            if isinstance(exog_vc, np.ndarray):
                exog_vc = exog_vc[:, None]
            else:
                exog_vc = pd.DataFrame(exog_vc)
        if exog_vc.ndim != 2:
            msg = "'exog_vc' must have one or two columns"
            raise ValueError(msg)
        ident = np.asarray(ident)
        if ident.ndim != 1:
            msg = 'ident must be a one-dimensional array'
            raise ValueError(msg)
        if len(ident) != exog_vc.shape[1]:
            msg = 'len(ident) should match the number of columns of exog_vc'
            raise ValueError(msg)
        if not np.issubdtype(ident.dtype, np.integer):
            msg = 'ident must have an integer dtype'
            raise ValueError(msg)
        if fep_names is None:
            if hasattr(exog, 'columns'):
                fep_names = exog.columns.tolist()
            else:
                fep_names = ['FE_%d' % (k + 1) for k in range(exog.shape[1])]
        if vcp_names is None:
            vcp_names = ['VC_%d' % (k + 1) for k in range(int(max(ident)) + 1)]
        elif len(vcp_names) != len(set(ident)):
            msg = 'The lengths of vcp_names and ident should be the same'
            raise ValueError(msg)
        if not sparse.issparse(exog_vc):
            exog_vc = sparse.csr_matrix(exog_vc)
        ident = ident.astype(int)
        vcp_p = float(vcp_p)
        fe_p = float(fe_p)
        if exog is None:
            k_fep = 0
        else:
            k_fep = exog.shape[1]
        if exog_vc is None:
            k_vc = 0
            k_vcp = 0
        else:
            k_vc = exog_vc.shape[1]
            k_vcp = max(ident) + 1
        exog_vc2 = exog_vc.multiply(exog_vc)
        super().__init__(endog, exog, **kwargs)
        self.exog_vc = exog_vc
        self.exog_vc2 = exog_vc2
        self.ident = ident
        self.family = family
        self.k_fep = k_fep
        self.k_vc = k_vc
        self.k_vcp = k_vcp
        self.fep_names = fep_names
        self.vcp_names = vcp_names
        self.vc_names = vc_names
        self.fe_p = fe_p
        self.vcp_p = vcp_p
        self.names = fep_names + vcp_names
        if vc_names is not None:
            self.names += vc_names

    def _unpack(self, vec):
        ii = 0
        fep = vec[:ii + self.k_fep]
        ii += self.k_fep
        vcp = vec[ii:ii + self.k_vcp]
        ii += self.k_vcp
        vc = vec[ii:]
        return (fep, vcp, vc)

    def logposterior(self, params):
        """
        The overall log-density: log p(y, fe, vc, vcp).

        This differs by an additive constant from the log posterior
        log p(fe, vc, vcp | y).
        """
        fep, vcp, vc = self._unpack(params)
        lp = 0
        if self.k_fep > 0:
            lp += np.dot(self.exog, fep)
        if self.k_vc > 0:
            lp += self.exog_vc.dot(vc)
        mu = self.family.link.inverse(lp)
        ll = self.family.loglike(self.endog, mu)
        if self.k_vc > 0:
            vcp0 = vcp[self.ident]
            s = np.exp(vcp0)
            ll -= 0.5 * np.sum(vc ** 2 / s ** 2) + np.sum(vcp0)
            ll -= 0.5 * np.sum(vcp ** 2 / self.vcp_p ** 2)
        if self.k_fep > 0:
            ll -= 0.5 * np.sum(fep ** 2 / self.fe_p ** 2)
        return ll

    def logposterior_grad(self, params):
        """
        The gradient of the log posterior.
        """
        fep, vcp, vc = self._unpack(params)
        lp = 0
        if self.k_fep > 0:
            lp += np.dot(self.exog, fep)
        if self.k_vc > 0:
            lp += self.exog_vc.dot(vc)
        mu = self.family.link.inverse(lp)
        score_factor = (self.endog - mu) / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)
        te = [None, None, None]
        if self.k_fep > 0:
            te[0] = np.dot(score_factor, self.exog)
        if self.k_vc > 0:
            te[2] = self.exog_vc.transpose().dot(score_factor)
        if self.k_vc > 0:
            vcp0 = vcp[self.ident]
            s = np.exp(vcp0)
            u = vc ** 2 / s ** 2 - 1
            te[1] = np.bincount(self.ident, weights=u)
            te[2] -= vc / s ** 2
            te[1] -= vcp / self.vcp_p ** 2
        if self.k_fep > 0:
            te[0] -= fep / self.fe_p ** 2
        te = [x for x in te if x is not None]
        return np.concatenate(te)

    def _get_start(self):
        start_fep = np.zeros(self.k_fep)
        start_vcp = np.ones(self.k_vcp)
        start_vc = np.random.normal(size=self.k_vc)
        start = np.concatenate((start_fep, start_vcp, start_vc))
        return start

    @classmethod
    def from_formula(cls, formula, vc_formulas, data, family=None, vcp_p=1, fe_p=2):
        """
        Fit a BayesMixedGLM using a formula.

        Parameters
        ----------
        formula : str
            Formula for the endog and fixed effects terms (use ~ to
            separate dependent and independent expressions).
        vc_formulas : dictionary
            vc_formulas[name] is a one-sided formula that creates one
            collection of random effects with a common variance
            parameter.  If using categorical (factor) variables to
            produce variance components, note that generally `0 + ...`
            should be used so that an intercept is not included.
        data : data frame
            The data to which the formulas are applied.
        family : genmod.families instance
            A GLM family.
        vcp_p : float
            The prior standard deviation for the logarithms of the standard
            deviations of the random effects.
        fe_p : float
            The prior standard deviation for the fixed effects parameters.
        """
        ident = []
        exog_vc = []
        vcp_names = []
        j = 0
        for na, fml in vc_formulas.items():
            mat = patsy.dmatrix(fml, data, return_type='dataframe')
            exog_vc.append(mat)
            vcp_names.append(na)
            ident.append(j * np.ones(mat.shape[1], dtype=np.int_))
            j += 1
        exog_vc = pd.concat(exog_vc, axis=1)
        vc_names = exog_vc.columns.tolist()
        ident = np.concatenate(ident)
        model = super().from_formula(formula, data=data, family=family, subset=None, exog_vc=exog_vc, ident=ident, vc_names=vc_names, vcp_names=vcp_names, fe_p=fe_p, vcp_p=vcp_p)
        return model

    def fit(self, method='BFGS', minim_opts=None):
        """
        fit is equivalent to fit_map.

        See fit_map for parameter information.

        Use `fit_vb` to fit the model using variational Bayes.
        """
        self.fit_map(method, minim_opts)

    def fit_map(self, method='BFGS', minim_opts=None, scale_fe=False):
        """
        Construct the Laplace approximation to the posterior distribution.

        Parameters
        ----------
        method : str
            Optimization method for finding the posterior mode.
        minim_opts : dict
            Options passed to scipy.minimize.
        scale_fe : bool
            If True, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.

        Returns
        -------
        BayesMixedGLMResults instance.
        """
        if scale_fe:
            mn = self.exog.mean(0)
            sc = self.exog.std(0)
            self._exog_save = self.exog
            self.exog = self.exog.copy()
            ixs = np.flatnonzero(sc > 1e-08)
            self.exog[:, ixs] -= mn[ixs]
            self.exog[:, ixs] /= sc[ixs]

        def fun(params):
            return -self.logposterior(params)

        def grad(params):
            return -self.logposterior_grad(params)
        start = self._get_start()
        r = minimize(fun, start, method=method, jac=grad, options=minim_opts)
        if not r.success:
            msg = 'Laplace fitting did not converge, |gradient|=%.6f' % np.sqrt(np.sum(r.jac ** 2))
            warnings.warn(msg)
        from statsmodels.tools.numdiff import approx_fprime
        hess = approx_fprime(r.x, grad)
        cov = np.linalg.inv(hess)
        params = r.x
        if scale_fe:
            self.exog = self._exog_save
            del self._exog_save
            params[ixs] /= sc[ixs]
            cov[ixs, :][:, ixs] /= np.outer(sc[ixs], sc[ixs])
        return BayesMixedGLMResults(self, params, cov, optim_retvals=r)

    def predict(self, params, exog=None, linear=False):
        """
        Return the fitted mean structure.

        Parameters
        ----------
        params : array_like
            The parameter vector, may be the full parameter vector, or may
            be truncated to include only the mean parameters.
        exog : array_like
            The design matrix for the mean structure.  If omitted, use the
            model's design matrix.
        linear : bool
            If True, return the linear predictor without passing through the
            link function.

        Returns
        -------
        A 1-dimensional array of predicted values
        """
        if exog is None:
            exog = self.exog
        q = exog.shape[1]
        pr = np.dot(exog, params[0:q])
        if not linear:
            pr = self.family.link.inverse(pr)
        return pr