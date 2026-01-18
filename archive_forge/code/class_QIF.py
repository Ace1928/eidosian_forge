import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
class QIF(base.Model):
    """
    Fit a regression model using quadratic inference functions (QIF).

    QIF is an alternative to GEE that can be more efficient, and that
    offers different approaches for model selection and inference.

    Parameters
    ----------
    endog : array_like
        The dependent variables of the regression.
    exog : array_like
        The independent variables of the regression.
    groups : array_like
        Labels indicating which group each observation belongs to.
        Observations in different groups should be independent.
    family : genmod family
        An instance of a GLM family.\x7f
    cov_struct : QIFCovariance instance
        An instance of a QIFCovariance.

    References
    ----------
    A. Qu, B. Lindsay, B. Li (2000).  Improving Generalized Estimating
    Equations using Quadratic Inference Functions, Biometrika 87:4.
    www.jstor.org/stable/2673612
    """

    def __init__(self, endog, exog, groups, family=None, cov_struct=None, missing='none', **kwargs):
        if family is None:
            family = families.Gaussian()
        elif not issubclass(family.__class__, families.Family):
            raise ValueError('QIF: `family` must be a genmod family instance')
        self.family = family
        self._fit_history = defaultdict(list)
        if cov_struct is None:
            cov_struct = QIFIndependence()
        elif not isinstance(cov_struct, QIFCovariance):
            raise ValueError('QIF: `cov_struct` must be a QIFCovariance instance')
        self.cov_struct = cov_struct
        groups = np.asarray(groups)
        super().__init__(endog, exog, groups=groups, missing=missing, **kwargs)
        self.group_names = list(set(groups))
        self.nobs = len(self.endog)
        groups_ix = defaultdict(list)
        for i, g in enumerate(groups):
            groups_ix[g].append(i)
        self.groups_ix = [groups_ix[na] for na in self.group_names]
        self._check_args(groups)

    def _check_args(self, groups):
        if len(groups) != len(self.endog):
            msg = 'QIF: groups and endog should have the same length'
            raise ValueError(msg)
        if len(self.endog) != self.exog.shape[0]:
            msg = 'QIF: the length of endog should be equal to the number of rows of exog.'
            raise ValueError(msg)

    def objective(self, params):
        """
        Calculate the gradient of the QIF objective function.

        Parameters
        ----------
        params : array_like
            The model parameters at which the gradient is evaluated.

        Returns
        -------
        grad : array_like
            The gradient vector of the QIF objective function.
        gn_deriv : array_like
            The gradients of each estimating equation with
            respect to the parameter.
        """
        endog = self.endog
        exog = self.exog
        lpr = np.dot(exog, params)
        mean = self.family.link.inverse(lpr)
        va = self.family.variance(mean)
        idl = self.family.link.inverse_deriv(lpr)
        idl2 = self.family.link.inverse_deriv2(lpr)
        vd = self.family.variance.deriv(mean)
        m = self.cov_struct.num_terms
        p = exog.shape[1]
        d = p * m
        gn = np.zeros(d)
        gi = np.zeros(d)
        gi_deriv = np.zeros((d, p))
        gn_deriv = np.zeros((d, p))
        cn_deriv = [0] * p
        cmat = np.zeros((d, d))
        fastvar = self.family.variance is varfuncs.constant
        fastlink = isinstance(self.family.link, (links.Identity, links.identity))
        for ix in self.groups_ix:
            sd = np.sqrt(va[ix])
            resid = endog[ix] - mean[ix]
            sresid = resid / sd
            deriv = exog[ix, :] * idl[ix, None]
            jj = 0
            for j in range(m):
                c = self.cov_struct.mat(len(ix), j)
                crs1 = np.dot(c, sresid) / sd
                gi[jj:jj + p] = np.dot(deriv.T, crs1)
                crs2 = np.dot(c, -deriv / sd[:, None]) / sd[:, None]
                gi_deriv[jj:jj + p, :] = np.dot(deriv.T, crs2)
                if not (fastlink and fastvar):
                    for k in range(p):
                        m1 = np.dot(exog[ix, :].T, idl2[ix] * exog[ix, k] * crs1)
                        if not fastvar:
                            vx = -0.5 * vd[ix] * deriv[:, k] / va[ix] ** 1.5
                            m2 = np.dot(deriv.T, vx * np.dot(c, sresid))
                            m3 = np.dot(deriv.T, np.dot(c, vx * resid) / sd)
                        else:
                            m2, m3 = (0, 0)
                        gi_deriv[jj:jj + p, k] += m1 + m2 + m3
                jj += p
            for j in range(p):
                u = np.outer(gi, gi_deriv[:, j])
                cn_deriv[j] += u + u.T
            gn += gi
            gn_deriv += gi_deriv
            cmat += np.outer(gi, gi)
        ngrp = len(self.groups_ix)
        gn /= ngrp
        gn_deriv /= ngrp
        cmat /= ngrp ** 2
        qif = np.dot(gn, np.linalg.solve(cmat, gn))
        gcg = np.zeros(p)
        for j in range(p):
            cn_deriv[j] /= len(self.groups_ix) ** 2
            u = np.linalg.solve(cmat, cn_deriv[j]).T
            u = np.linalg.solve(cmat, u)
            gcg[j] = np.dot(gn, np.dot(u, gn))
        grad = 2 * np.dot(gn_deriv.T, np.linalg.solve(cmat, gn)) - gcg
        return (qif, grad, cmat, gn, gn_deriv)

    def estimate_scale(self, params):
        """
        Estimate the dispersion/scale.

        The scale parameter for binomial and Poisson families is
        fixed at 1, otherwise it is estimated from the data.
        """
        if isinstance(self.family, (families.Binomial, families.Poisson)):
            return 1.0
        if hasattr(self, 'ddof_scale'):
            ddof_scale = self.ddof_scale
        else:
            ddof_scale = self.exog[1]
        lpr = np.dot(self.exog, params)
        mean = self.family.link.inverse(lpr)
        resid = self.endog - mean
        scale = np.sum(resid ** 2) / (self.nobs - ddof_scale)
        return scale

    @classmethod
    def from_formula(cls, formula, groups, data, subset=None, *args, **kwargs):
        """
        Create a QIF model instance from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        groups : array_like or string
            Array of grouping labels.  If a string, this is the name
            of a variable in `data` that contains the grouping labels.
        data : array_like
            The data for the model.
        subset : array_like
            An array_like object of booleans, integers, or index
            values that indicate the subset of the data to used when
            fitting the model.

        Returns
        -------
        model : QIF model instance
        """
        if isinstance(groups, str):
            groups = data[groups]
        model = super().from_formula(formula, *args, data=data, subset=subset, groups=groups, **kwargs)
        return model

    def fit(self, maxiter=100, start_params=None, tol=1e-06, gtol=0.0001, ddof_scale=None):
        """
        Fit a GLM to correlated data using QIF.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        start_params : array_like, optional
            Starting values
        tol : float
            Convergence threshold for difference of successive
            estimates.
        gtol : float
            Convergence threshold for gradient.
        ddof_scale : int, optional
            Degrees of freedom for the scale parameter

        Returns
        -------
        QIFResults object
        """
        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            self.ddof_scale = ddof_scale
        if start_params is None:
            model = GLM(self.endog, self.exog, family=self.family)
            result = model.fit()
            params = result.params
        else:
            params = start_params
        for _ in range(maxiter):
            qif, grad, cmat, _, gn_deriv = self.objective(params)
            gnorm = np.sqrt(np.sum(grad * grad))
            self._fit_history['qif'].append(qif)
            self._fit_history['gradnorm'].append(gnorm)
            if gnorm < gtol:
                break
            cjac = 2 * np.dot(gn_deriv.T, np.linalg.solve(cmat, gn_deriv))
            step = np.linalg.solve(cjac, grad)
            snorm = np.sqrt(np.sum(step * step))
            self._fit_history['stepnorm'].append(snorm)
            if snorm < tol:
                break
            params -= step
        vcov = np.dot(gn_deriv.T, np.linalg.solve(cmat, gn_deriv))
        vcov = np.linalg.inv(vcov)
        scale = self.estimate_scale(params)
        rslt = QIFResults(self, params, vcov / scale, scale)
        rslt.fit_history = self._fit_history
        self._fit_history = defaultdict(list)
        return QIFResultsWrapper(rslt)