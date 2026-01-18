import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
class TTestPower(Power):
    """Statistical Power calculations for one sample or paired sample t-test

    """

    def power(self, effect_size, nobs, alpha, df=None, alternative='two-sided'):
        """Calculate the power of a t-test for one sample or paired samples.

        Parameters
        ----------
        effect_size : float
            standardized effect size, mean divided by the standard deviation.
            effect size has to be positive.
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        df : int or float
            degrees of freedom. By default this is None, and the df from the
            one sample or paired ttest is used, ``df = nobs1 - 1``
        alternative : str, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.
            .

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       """
        return ttest_power(effect_size, nobs, alpha, df=df, alternative=alternative)

    def solve_power(self, effect_size=None, nobs=None, alpha=None, power=None, alternative='two-sided'):
        """solve for any one parameter of the power of a one sample t-test

        for the one sample t-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        This test can also be used for a paired t-test, where effect size is
        defined in terms of the mean difference, and nobs is the number of
        pairs.

        Parameters
        ----------
        effect_size : float
            Standardized effect size.The effect size is here Cohen's f, square
            root of "f2".
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        alternative : str, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.

        *attaches*

        cache_fit_res : list
            Cache of the result of the root finding procedure for the latest
            call to ``solve_power``, mainly for debugging purposes.
            The first element is the success indicator, one if successful.
            The remaining elements contain the return information of the up to
            three solvers that have been tried.

        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        """
        return super().solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, power=power, alternative=alternative)