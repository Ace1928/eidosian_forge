import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
class DescStatUV(_OptFuncts):
    """
    A class to compute confidence intervals and hypothesis tests involving
    mean, variance, kurtosis and skewness of a univariate random variable.

    Parameters
    ----------
    endog : 1darray
        Data to be analyzed

    Attributes
    ----------
    endog : 1darray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        self.endog = np.squeeze(endog)
        self.nobs = endog.shape[0]

    def test_mean(self, mu0, return_weights=False):
        """
        Returns - 2 x log-likelihood ratio, p-value and weights
        for a hypothesis test of the mean.

        Parameters
        ----------
        mu0 : float
            Mean value to be tested

        return_weights : bool
            If return_weights is True the function returns
            the weights of the observations under the null hypothesis.
            Default is False

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of mu0
        """
        self.mu0 = mu0
        endog = self.endog
        nobs = self.nobs
        eta_min = (1.0 - 1.0 / nobs) / (self.mu0 - max(endog))
        eta_max = (1.0 - 1.0 / nobs) / (self.mu0 - min(endog))
        eta_star = optimize.brentq(self._find_eta, eta_min, eta_max)
        new_weights = 1.0 / nobs * 1.0 / (1.0 + eta_star * (endog - self.mu0))
        llr = -2 * np.sum(np.log(nobs * new_weights))
        if return_weights:
            return (llr, chi2.sf(llr, 1), new_weights)
        else:
            return (llr, chi2.sf(llr, 1))

    def ci_mean(self, sig=0.05, method='gamma', epsilon=10 ** (-8), gamma_low=-10 ** 10, gamma_high=10 ** 10):
        """
        Returns the confidence interval for the mean.

        Parameters
        ----------
        sig : float
            significance level. Default is .05

        method : str
            Root finding method,  Can be 'nested-brent' or
            'gamma'.  Default is 'gamma'

            'gamma' Tries to solve for the gamma parameter in the
            Lagrange (see Owen pg 22) and then determine the weights.

            'nested brent' uses brents method to find the confidence
            intervals but must maximize the likelihood ratio on every
            iteration.

            gamma is generally much faster.  If the optimizations does not
            converge, try expanding the gamma_high and gamma_low
            variable.

        gamma_low : float
            Lower bound for gamma when finding lower limit.
            If function returns f(a) and f(b) must have different signs,
            consider lowering gamma_low.

        gamma_high : float
            Upper bound for gamma when finding upper limit.
            If function returns f(a) and f(b) must have different signs,
            consider raising gamma_high.

        epsilon : float
            When using 'nested-brent', amount to decrease (increase)
            from the maximum (minimum) of the data when
            starting the search.  This is to protect against the
            likelihood ratio being zero at the maximum (minimum)
            value of the data.  If data is very small in absolute value
            (<10 ``**`` -6) consider shrinking epsilon

            When using 'gamma', amount to decrease (increase) the
            minimum (maximum) by to start the search for gamma.
            If function returns f(a) and f(b) must have different signs,
            consider lowering epsilon.

        Returns
        -------
        Interval : tuple
            Confidence interval for the mean
        """
        endog = self.endog
        sig = 1 - sig
        if method == 'nested-brent':
            self.r0 = chi2.ppf(sig, 1)
            middle = np.mean(endog)
            epsilon_u = (max(endog) - np.mean(endog)) * epsilon
            epsilon_l = (np.mean(endog) - min(endog)) * epsilon
            ulim = optimize.brentq(self._ci_limits_mu, middle, max(endog) - epsilon_u)
            llim = optimize.brentq(self._ci_limits_mu, middle, min(endog) + epsilon_l)
            return (llim, ulim)
        if method == 'gamma':
            self.r0 = chi2.ppf(sig, 1)
            gamma_star_l = optimize.brentq(self._find_gamma, gamma_low, min(endog) - epsilon)
            gamma_star_u = optimize.brentq(self._find_gamma, max(endog) + epsilon, gamma_high)
            weights_low = (endog - gamma_star_l) ** (-1) / np.sum((endog - gamma_star_l) ** (-1))
            weights_high = (endog - gamma_star_u) ** (-1) / np.sum((endog - gamma_star_u) ** (-1))
            mu_low = np.sum(weights_low * endog)
            mu_high = np.sum(weights_high * endog)
            return (mu_low, mu_high)

    def test_var(self, sig2_0, return_weights=False):
        """
        Returns  -2 x log-likelihood ratio and the p-value for the
        hypothesized variance

        Parameters
        ----------
        sig2_0 : float
            Hypothesized variance to be tested

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of observing sig2_0. Default is False

        Returns
        -------
        test_results : tuple
            The  log-likelihood ratio and the p_value  of sig2_0

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(1000)*100
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> hyp_test = el_analysis.test_var(9500)
        """
        self.sig2_0 = sig2_0
        mu_max = max(self.endog)
        mu_min = min(self.endog)
        llr = optimize.fminbound(self._opt_var, mu_min, mu_max, full_output=1)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        else:
            return (llr, p_val)

    def ci_var(self, lower_bound=None, upper_bound=None, sig=0.05):
        """
        Returns the confidence interval for the variance.

        Parameters
        ----------
        lower_bound : float
            The minimum value the lower confidence interval can
            take. The p-value from test_var(lower_bound) must be lower
            than 1 - significance level. Default is .99 confidence
            limit assuming normality

        upper_bound : float
            The maximum value the upper confidence interval
            can take. The p-value from test_var(upper_bound) must be lower
            than 1 - significance level.  Default is .99 confidence
            limit assuming normality

        sig : float
            The significance level. Default is .05

        Returns
        -------
        Interval : tuple
            Confidence interval for the variance

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(100)
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> el_analysis.ci_var()
        (0.7539322567470305, 1.229998852496268)
        >>> el_analysis.ci_var(.5, 2)
        (0.7539322567469926, 1.2299988524962664)

        Notes
        -----
        If the function returns the error f(a) and f(b) must have
        different signs, consider lowering lower_bound and raising
        upper_bound.
        """
        endog = self.endog
        if upper_bound is None:
            upper_bound = (self.nobs - 1) * endog.var() / chi2.ppf(0.0001, self.nobs - 1)
        if lower_bound is None:
            lower_bound = (self.nobs - 1) * endog.var() / chi2.ppf(0.9999, self.nobs - 1)
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_var, lower_bound, endog.var())
        ulim = optimize.brentq(self._ci_limits_var, endog.var(), upper_bound)
        return (llim, ulim)

    def plot_contour(self, mu_low, mu_high, var_low, var_high, mu_step, var_step, levs=[0.2, 0.1, 0.05, 0.01, 0.001]):
        """
        Returns a plot of the confidence region for a univariate
        mean and variance.

        Parameters
        ----------
        mu_low : float
            Lowest value of the mean to plot

        mu_high : float
            Highest value of the mean to plot

        var_low : float
            Lowest value of the variance to plot

        var_high : float
            Highest value of the variance to plot

        mu_step : float
            Increments to evaluate the mean

        var_step : float
            Increments to evaluate the mean

        levs : list
            Which values of significance the contour lines will be drawn.
            Default is [.2, .1, .05, .01, .001]

        Returns
        -------
        Figure
            The contour plot
        """
        fig, ax = utils.create_mpl_ax()
        ax.set_ylabel('Variance')
        ax.set_xlabel('Mean')
        mu_vect = list(np.arange(mu_low, mu_high, mu_step))
        var_vect = list(np.arange(var_low, var_high, var_step))
        z = []
        for sig0 in var_vect:
            self.sig2_0 = sig0
            for mu0 in mu_vect:
                z.append(self._opt_var(mu0, pval=True))
        z = np.asarray(z).reshape(len(var_vect), len(mu_vect))
        ax.contour(mu_vect, var_vect, z, levels=levs)
        return fig

    def test_skew(self, skew0, return_weights=False):
        """
        Returns  -2 x log-likelihood and p-value for the hypothesized
        skewness.

        Parameters
        ----------
        skew0 : float
            Skewness value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p_value of skew0
        """
        self.skew0 = skew0
        start_nuisance = np.array([self.endog.mean(), self.endog.var()])
        llr = optimize.fmin_powell(self._opt_skew, start_nuisance, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def test_kurt(self, kurt0, return_weights=False):
        """
        Returns -2 x log-likelihood and the p-value for the hypothesized
        kurtosis.

        Parameters
        ----------
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of kurt0
        """
        self.kurt0 = kurt0
        start_nuisance = np.array([self.endog.mean(), self.endog.var()])
        llr = optimize.fmin_powell(self._opt_kurt, start_nuisance, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def test_joint_skew_kurt(self, skew0, kurt0, return_weights=False):
        """
        Returns - 2 x log-likelihood and the p-value for the joint
        hypothesis test for skewness and kurtosis

        Parameters
        ----------
        skew0 : float
            Skewness value to be tested
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value  of the joint hypothesis test.
        """
        self.skew0 = skew0
        self.kurt0 = kurt0
        start_nuisance = np.array([self.endog.mean(), self.endog.var()])
        llr = optimize.fmin_powell(self._opt_skew_kurt, start_nuisance, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 2)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def ci_skew(self, sig=0.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence interval for skewness.

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of skewness the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of skewness the lower limit can be.
            Default is .99 confidence level assuming normality.

        Returns
        -------
        Interval : tuple
            Confidence interval for the skewness

        Notes
        -----
        If function returns f(a) and f(b) must have different signs, consider
        expanding lower and upper bounds
        """
        nobs = self.nobs
        endog = self.endog
        if upper_bound is None:
            upper_bound = skew(endog) + 2.5 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5
        if lower_bound is None:
            lower_bound = skew(endog) - 2.5 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_skew, lower_bound, skew(endog))
        ulim = optimize.brentq(self._ci_limits_skew, skew(endog), upper_bound)
        return (llim, ulim)

    def ci_kurt(self, sig=0.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence interval for kurtosis.

        Parameters
        ----------

        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of kurtosis the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of kurtosis the lower limit can be.
            Default is .99 confidence limit assuming normality.

        Returns
        -------
        Interval : tuple
            Lower and upper confidence limit

        Notes
        -----
        For small n, upper_bound and lower_bound may have to be
        provided by the user.  Consider using test_kurt to find
        values close to the desired significance level.

        If function returns f(a) and f(b) must have different signs, consider
        expanding the bounds.
        """
        endog = self.endog
        nobs = self.nobs
        if upper_bound is None:
            upper_bound = kurtosis(endog) + 2.5 * (2.0 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5) * ((nobs ** 2.0 - 1.0) / ((nobs - 3.0) * (nobs + 5.0))) ** 0.5
        if lower_bound is None:
            lower_bound = kurtosis(endog) - 2.5 * (2.0 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5) * ((nobs ** 2.0 - 1.0) / ((nobs - 3.0) * (nobs + 5.0))) ** 0.5
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_kurt, lower_bound, kurtosis(endog))
        ulim = optimize.brentq(self._ci_limits_kurt, kurtosis(endog), upper_bound)
        return (llim, ulim)