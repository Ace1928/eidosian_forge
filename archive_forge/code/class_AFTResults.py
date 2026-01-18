import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
class AFTResults(OptAFT):

    def __init__(self, model):
        self.model = model

    def params(self):
        """

        Fits an AFT model and returns parameters.

        Parameters
        ----------
        None


        Returns
        -------
        Fitted params

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        self.model.modif_censors = np.copy(self.model.censors)
        self.model.modif_censors[-1] = 1
        wts = self.model._make_km(self.model.endog, self.model.modif_censors)
        res = WLS(self.model.endog, self.model.exog, wts).fit()
        params = res.params
        return params

    def test_beta(self, b0_vals, param_nums, ftol=10 ** (-5), maxiter=30, print_weights=1):
        """
        Returns the profile log likelihood for regression parameters
        'param_num' at 'b0_vals.'

        Parameters
        ----------
        b0_vals : list
            The value of parameters to be tested
        param_num : list
            Which parameters to be tested
        maxiter : int, optional
            How many iterations to use in the EM algorithm.  Default is 30
        ftol : float, optional
            The function tolerance for the EM optimization.
            Default is 10''**''-5
        print_weights : bool
            If true, returns the weights tate maximize the profile
            log likelihood. Default is False

        Returns
        -------

        test_results : tuple
            The log-likelihood and p-pvalue of the test.

        Notes
        -----

        The function will warn if the EM reaches the maxiter.  However, when
        optimizing over nuisance parameters, it is possible to reach a
        maximum number of inner iterations for a specific value for the
        nuisance parameters while the resultsof the function are still valid.
        This usually occurs when the optimization over the nuisance parameters
        selects parameter values that yield a log-likihood ratio close to
        infinity.

        Examples
        --------

        >>> import statsmodels.api as sm
        >>> import numpy as np

        # Test parameter is .05 in one regressor no intercept model
        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, x, cens)
        >>> res=model.test_beta([0], [0])
        >>> res
        (1.4657739632606308, 0.22601365256959183)

        #Test slope is 0 in  model with intercept

        >>> data=sm.datasets.heart.load()
        >>> y = np.log10(data.endog)
        >>> x = data.exog
        >>> cens = data.censors
        >>> model = sm.emplike.emplikeAFT(y, sm.add_constant(x), cens)
        >>> res = model.test_beta([0], [1])
        >>> res
        (4.623487775078047, 0.031537049752572731)
        """
        censors = self.model.censors
        endog = self.model.endog
        exog = self.model.exog
        uncensored = (censors == 1).flatten()
        censored = (censors == 0).flatten()
        uncens_endog = endog[uncensored]
        uncens_exog = exog[uncensored, :]
        reg_model = OLS(uncens_endog, uncens_exog).fit()
        llr, pval, new_weights = reg_model.el_test(b0_vals, param_nums, return_weights=True)
        km = self.model._make_km(endog, censors).flatten()
        uncens_nobs = self.model.uncens_nobs
        F = np.asarray(new_weights).reshape(uncens_nobs)
        params = self.params()
        survidx = np.where(censors == 0)
        survidx = survidx[0] - np.arange(len(survidx[0]))
        numcensbelow = np.int_(np.cumsum(1 - censors))
        if len(param_nums) == len(params):
            llr = self._EM_test([], F=F, params=params, param_nums=param_nums, b0_vals=b0_vals, survidx=survidx, uncens_nobs=uncens_nobs, numcensbelow=numcensbelow, km=km, uncensored=uncensored, censored=censored, ftol=ftol, maxiter=25)
            return (llr, chi2.sf(llr, self.model.nvar))
        else:
            x0 = np.delete(params, param_nums)
            try:
                res = optimize.fmin(self._EM_test, x0, (params, param_nums, b0_vals, F, survidx, uncens_nobs, numcensbelow, km, uncensored, censored, maxiter, ftol), full_output=1, disp=0)
                llr = res[1]
                return (llr, chi2.sf(llr, len(param_nums)))
            except np.linalg.LinAlgError:
                return (np.inf, 0)

    def ci_beta(self, param_num, beta_high, beta_low, sig=0.05):
        """
        Returns the confidence interval for a regression
        parameter in the AFT model.

        Parameters
        ----------
        param_num : int
            Parameter number of interest
        beta_high : float
            Upper bound for the confidence interval
        beta_low : float
            Lower bound for the confidence interval
        sig : float, optional
            Significance level.  Default is .05

        Notes
        -----
        If the function returns f(a) and f(b) must have different signs,
        consider widening the search area by adjusting beta_low and
        beta_high.

        Also note that this process is computational intensive.  There
        are 4 levels of optimization/solving.  From outer to inner:

        1) Solving so that llr-critical value = 0
        2) maximizing over nuisance parameters
        3) Using  EM at each value of nuisamce parameters
        4) Using the _modified_Newton optimizer at each iteration
           of the EM algorithm.

        Also, for very unlikely nuisance parameters, it is possible for
        the EM algorithm to not converge.  This is not an indicator
        that the solver did not find the correct solution.  It just means
        for a specific iteration of the nuisance parameters, the optimizer
        was unable to converge.

        If the user desires to verify the success of the optimization,
        it is recommended to test the limits using test_beta.
        """
        params = self.params()
        self.r0 = chi2.ppf(1 - sig, 1)
        ll = optimize.brentq(self._ci_limits_beta, beta_low, params[param_num], param_num)
        ul = optimize.brentq(self._ci_limits_beta, params[param_num], beta_high, param_num)
        return (ll, ul)