import numpy as np
from statsmodels.stats._knockoff import RegressionFDR
class NullDistribution:
    """
    Estimate a Gaussian distribution for the null Z-scores.

    The observed Z-scores consist of both null and non-null values.
    The fitted distribution of null Z-scores is Gaussian, but may have
    non-zero mean and/or non-unit scale.

    Parameters
    ----------
    zscores : array_like
        The observed Z-scores.
    null_lb : float
        Z-scores between `null_lb` and `null_ub` are all considered to be
        true null hypotheses.
    null_ub : float
        See `null_lb`.
    estimate_mean : bool
        If True, estimate the mean of the distribution.  If False, the
        mean is fixed at zero.
    estimate_scale : bool
        If True, estimate the scale of the distribution.  If False, the
        scale parameter is fixed at 1.
    estimate_null_proportion : bool
        If True, estimate the proportion of true null hypotheses (i.e.
        the proportion of z-scores with expected value zero).  If False,
        this parameter is fixed at 1.

    Attributes
    ----------
    mean : float
        The estimated mean of the empirical null distribution
    sd : float
        The estimated standard deviation of the empirical null distribution
    null_proportion : float
        The estimated proportion of true null hypotheses among all hypotheses

    References
    ----------
    B Efron (2008).  Microarrays, Empirical Bayes, and the Two-Groups
    Model.  Statistical Science 23:1, 1-22.

    Notes
    -----
    See also:

    http://nipy.org/nipy/labs/enn.html#nipy.algorithms.statistics.empirical_pvalue.NormalEmpiricalNull.fdr
    """

    def __init__(self, zscores, null_lb=-1, null_ub=1, estimate_mean=True, estimate_scale=True, estimate_null_proportion=False):
        ii = np.flatnonzero((zscores >= null_lb) & (zscores <= null_ub))
        if len(ii) == 0:
            raise RuntimeError('No Z-scores fall between null_lb and null_ub')
        zscores0 = zscores[ii]
        n_zs, n_zs0 = (len(zscores), len(zscores0))

        def xform(params):
            mean = 0.0
            sd = 1.0
            prob = 1.0
            ii = 0
            if estimate_mean:
                mean = params[ii]
                ii += 1
            if estimate_scale:
                sd = np.exp(params[ii])
                ii += 1
            if estimate_null_proportion:
                prob = 1 / (1 + np.exp(-params[ii]))
            return (mean, sd, prob)
        from scipy.stats.distributions import norm

        def fun(params):
            """
            Negative log-likelihood of z-scores.

            The function has three arguments, packed into a vector:

            mean : location parameter
            logscale : log of the scale parameter
            logitprop : logit of the proportion of true nulls

            The implementation follows section 4 from Efron 2008.
            """
            d, s, p = xform(params)
            central_mass = norm.cdf((null_ub - d) / s) - norm.cdf((null_lb - d) / s)
            cp = p * central_mass
            rval = n_zs0 * np.log(cp) + (n_zs - n_zs0) * np.log(1 - cp)
            zv = (zscores0 - d) / s
            rval += np.sum(-zv ** 2 / 2) - n_zs0 * np.log(s)
            rval -= n_zs0 * np.log(central_mass)
            return -rval
        from scipy.optimize import minimize
        mz = minimize(fun, np.r_[0.0, 0, 3], method='Nelder-Mead')
        mean, sd, prob = xform(mz['x'])
        self.mean = mean
        self.sd = sd
        self.null_proportion = prob

    def pdf(self, zscores):
        """
        Evaluates the fitted empirical null Z-score density.

        Parameters
        ----------
        zscores : scalar or array_like
            The point or points at which the density is to be
            evaluated.

        Returns
        -------
        The empirical null Z-score density evaluated at the given
        points.
        """
        zval = (zscores - self.mean) / self.sd
        return np.exp(-0.5 * zval ** 2 - np.log(self.sd) - 0.5 * np.log(2 * np.pi))