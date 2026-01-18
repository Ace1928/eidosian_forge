import numpy as np
from .descriptive import _OptFuncts
from scipy import optimize
from scipy.stats import chi2
class ANOVA(_ANOVAOpt):
    """
    A class for ANOVA and comparing means.

    Parameters
    ----------

    endog : list of arrays
        endog should be a list containing 1 dimensional arrays.  Each array
        is the data collected from a certain group.
    """

    def __init__(self, endog):
        self.endog = endog
        self.num_groups = len(self.endog)
        self.nobs = 0
        for i in self.endog:
            self.nobs = self.nobs + len(i)

    def compute_ANOVA(self, mu=None, mu_start=0, return_weights=0):
        """
        Returns -2 log likelihood, the pvalue and the maximum likelihood
        estimate for a common mean.

        Parameters
        ----------

        mu : float
            If a mu is specified, ANOVA is conducted with mu as the
            common mean.  Otherwise, the common mean is the maximum
            empirical likelihood estimate of the common mean.
            Default is None.

        mu_start : float
            Starting value for commean mean if specific mu is not specified.
            Default = 0.

        return_weights : bool
            if TRUE, returns the weights on observations that maximize the
            likelihood.  Default is FALSE.

        Returns
        -------

        res: tuple
            The log-likelihood, p-value and estimate for the common mean.
        """
        if mu is not None:
            llr = self._opt_common_mu(mu)
            pval = 1 - chi2.cdf(llr, self.num_groups - 1)
            if return_weights:
                return (llr, pval, mu, self.new_weights)
            else:
                return (llr, pval, mu)
        else:
            res = optimize.fmin_powell(self._opt_common_mu, mu_start, full_output=1, disp=False)
            llr = res[1]
            mu_common = float(np.squeeze(res[0]))
            pval = 1 - chi2.cdf(llr, self.num_groups - 1)
            if return_weights:
                return (llr, pval, mu_common, self.new_weights)
            else:
                return (llr, pval, mu_common)