from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class PoissonResults(CountResults):

    def predict_prob(self, n=None, exog=None, exposure=None, offset=None, transform=True):
        """
        Return predicted probability of each count level for each observation

        Parameters
        ----------
        n : array_like or int
            The counts for which you want the probabilities. If n is None
            then the probabilities for each count from 0 to max(y) are
            given.

        Returns
        -------
        ndarray
            A nobs x n array where len(`n`) columns are indexed by the count
            n. If n is None, then column 0 is the probability that each
            observation is 0, column 1 is the probability that each
            observation is 1, etc.
        """
        if n is not None:
            counts = np.atleast_2d(n)
        else:
            counts = np.atleast_2d(np.arange(0, np.max(self.model.endog) + 1))
        mu = self.predict(exog=exog, exposure=exposure, offset=offset, transform=transform, which='mean')[:, None]
        return stats.poisson.pmf(counts, mu)

    @property
    def resid_pearson(self):
        """
        Pearson residuals

        Notes
        -----
        Pearson residuals are defined to be

        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}

        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        p = self.predict()
        return (self.model.endog - p) / np.sqrt(p)

    def get_influence(self):
        """
        Get an instance of MLEInfluence with influence and outlier measures

        Returns
        -------
        infl : MLEInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence
        """
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def get_diagnostic(self, y_max=None):
        """
        Get instance of class with specification and diagnostic methods

        experimental, API of Diagnostic classes will change

        Returns
        -------
        PoissonDiagnostic instance
            The instance has methods to perform specification and diagnostic
            tesst and plots

        See Also
        --------
        statsmodels.statsmodels.discrete.diagnostic.PoissonDiagnostic
        """
        from statsmodels.discrete.diagnostic import PoissonDiagnostic
        return PoissonDiagnostic(self, y_max=y_max)