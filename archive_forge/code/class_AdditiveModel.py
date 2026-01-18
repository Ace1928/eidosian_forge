import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings
class AdditiveModel:
    """additive model with non-parametric, smoothed components

    Parameters
    ----------
    exog : ndarray
    smoothers : None or list of smoother instances
        smoother instances not yet checked
    weights : None or ndarray
    family : None or family instance
        I think only used because of shared results with GAM and subclassing.
        If None, then Gaussian is used.
    """

    def __init__(self, exog, smoothers=None, weights=None, family=None):
        self.exog = exog
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.exog.shape[0])
        self.smoothers = smoothers or [default_smoother(exog[:, i]) for i in range(exog.shape[1])]
        for i in range(exog.shape[1]):
            self.smoothers[i].df = 10
        if family is None:
            self.family = families.Gaussian()
        else:
            self.family = family

    def _iter__(self):
        """initialize iteration ?, should be removed

        """
        self.iter = 0
        self.dev = np.inf
        return self

    def next(self):
        """internal calculation for one fit iteration

        BUG: I think this does not improve, what is supposed to improve
            offset does not seem to be used, neither an old alpha
            The smoothers keep coef/params from previous iteration
        """
        _results = self.results
        Y = self.results.Y
        mu = _results.predict(self.exog)
        offset = np.zeros(self.exog.shape[1], np.float64)
        alpha = (Y * self.weights).sum() / self.weights.sum()
        for i in range(self.exog.shape[1]):
            tmp = self.smoothers[i].predict()
            bad = np.isnan(Y - alpha - mu + tmp).any()
            if bad:
                print(Y, alpha, mu, tmp)
                raise ValueError('nan encountered')
            self.smoothers[i].smooth(Y - mu + tmp, weights=self.weights)
            tmp2 = self.smoothers[i].predict()
            self.results.offset[i] = -(tmp2 * self.weights).sum() / self.weights.sum()
            if DEBUG:
                print(self.smoothers[i].params)
            mu += tmp2 - tmp
        offset = self.results.offset
        return Results(Y, alpha, self.exog, self.smoothers, self.family, offset)

    def cont(self):
        """condition to continue iteration loop

        Parameters
        ----------
        tol

        Returns
        -------
        cont : bool
            If true, then iteration should be continued.

        """
        self.iter += 1
        if DEBUG:
            print(self.iter, self.results.Y.shape)
            print(self.results.predict(self.exog).shape, self.weights.shape)
        curdev = ((self.results.Y - self.results.predict(self.exog)) ** 2 * self.weights).sum()
        if self.iter > self.maxiter:
            return False
        if np.fabs((self.dev - curdev) / curdev) < self.rtol:
            self.dev = curdev
            return False
        self.dev = curdev
        return True

    def df_resid(self):
        """degrees of freedom of residuals, ddof is sum of all smoothers df
        """
        return self.results.Y.shape[0] - np.array([self.smoothers[i].df_fit() for i in range(self.exog.shape[1])]).sum()

    def estimate_scale(self):
        """estimate standard deviation of residuals
        """
        return ((self.results.Y - self.results(self.exog)) ** 2).sum() / self.df_resid()

    def fit(self, Y, rtol=1e-06, maxiter=30):
        """fit the model to a given endogenous variable Y

        This needs to change for consistency with statsmodels

        """
        self.rtol = rtol
        self.maxiter = maxiter
        self._iter__()
        mu = 0
        alpha = (Y * self.weights).sum() / self.weights.sum()
        offset = np.zeros(self.exog.shape[1], np.float64)
        for i in range(self.exog.shape[1]):
            self.smoothers[i].smooth(Y - alpha - mu, weights=self.weights)
            tmp = self.smoothers[i].predict()
            offset[i] = (tmp * self.weights).sum() / self.weights.sum()
            tmp -= tmp.sum()
            mu += tmp
        self.results = Results(Y, alpha, self.exog, self.smoothers, self.family, offset)
        while self.cont():
            self.results = self.next()
        if self.iter >= self.maxiter:
            warnings.warn(iteration_limit_doc, IterationLimitWarning)
        return self.results