import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
class _OptFuncts:
    """
    A class that holds functions that are optimized/solved.

    The general setup of the class is simple.  Any method that starts with
    _opt_ creates a vector of estimating equations named est_vect such that
    np.dot(p, (est_vect))=0 where p is the weight on each
    observation as a 1 x n array and est_vect is n x k.  Then _modif_Newton is
    called to determine the optimal p by solving for the Lagrange multiplier
    (eta) in the profile likelihood maximization problem.  In the presence
    of nuisance parameters, _opt_ functions are  optimized over to profile
    out the nuisance parameters.

    Any method starting with _ci_limits calculates the log likelihood
    ratio for a specific value of a parameter and then subtracts a
    pre-specified critical value.  This is solved so that llr - crit = 0.
    """

    def __init__(self, endog):
        pass

    def _log_star(self, eta, est_vect, weights, nobs):
        """
        Transforms the log of observation probabilities in terms of the
        Lagrange multiplier to the log 'star' of the probabilities.

        Parameters
        ----------
        eta : float
            Lagrange multiplier

        est_vect : ndarray (n,k)
            Estimating equations vector

        wts : nx1 array
            Observation weights

        Returns
        ------
        data_star : ndarray
            The weighted logstar of the estimting equations

        Notes
        -----
        This function is only a placeholder for the _fit_Newton.
        The function value is not used in optimization and the optimal value
        is disregarded when computing the log likelihood ratio.
        """
        data_star = np.log(weights) + (np.sum(weights) + np.dot(est_vect, eta))
        idx = data_star < 1.0 / nobs
        not_idx = ~idx
        nx = nobs * data_star[idx]
        data_star[idx] = np.log(1.0 / nobs) - 1.5 + nx * (2.0 - nx / 2)
        data_star[not_idx] = np.log(data_star[not_idx])
        return data_star

    def _hess(self, eta, est_vect, weights, nobs):
        """
        Calculates the hessian of a weighted empirical likelihood
        problem.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        hess : m x m array
            Weighted hessian used in _wtd_modif_newton
        """
        data_star_doub_prime = np.sum(weights) + np.dot(est_vect, eta)
        idx = data_star_doub_prime < 1.0 / nobs
        not_idx = ~idx
        data_star_doub_prime[idx] = -nobs ** 2
        data_star_doub_prime[not_idx] = -data_star_doub_prime[not_idx] ** (-2)
        wtd_dsdp = weights * data_star_doub_prime
        return np.dot(est_vect.T, wtd_dsdp[:, None] * est_vect)

    def _grad(self, eta, est_vect, weights, nobs):
        """
        Calculates the gradient of a weighted empirical likelihood
        problem

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        gradient : ndarray (m,1)
            The gradient used in _wtd_modif_newton
        """
        data_star_prime = np.sum(weights) + np.dot(est_vect, eta)
        idx = data_star_prime < 1.0 / nobs
        not_idx = ~idx
        data_star_prime[idx] = nobs * (2 - nobs * data_star_prime[idx])
        data_star_prime[not_idx] = 1.0 / data_star_prime[not_idx]
        return np.dot(weights * data_star_prime, est_vect)

    def _modif_newton(self, eta, est_vect, weights):
        """
        Modified Newton's method for maximizing the log 'star' equation.  This
        function calls _fit_newton to find the optimal values of eta.

        Parameters
        ----------
        eta : ndarray, (1,m)
            Lagrange multiplier in the profile likelihood maximization

        est_vect : ndarray, (n,k)
            Estimating equations vector

        weights : 1darray
            Observation weights

        Returns
        -------
        params : 1xm array
            Lagrange multiplier that maximizes the log-likelihood
        """
        nobs = len(est_vect)
        f = lambda x0: -np.sum(self._log_star(x0, est_vect, weights, nobs))
        grad = lambda x0: -self._grad(x0, est_vect, weights, nobs)
        hess = lambda x0: -self._hess(x0, est_vect, weights, nobs)
        kwds = {'tol': 1e-08}
        eta = eta.squeeze()
        res = _fit_newton(f, grad, eta, (), kwds, hess=hess, maxiter=50, disp=0)
        return res[0]

    def _find_eta(self, eta):
        """
        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
        eta when computing ELR for univariate mean.

        Parameters
        ----------
        eta : float
            Lagrange multiplier in the empirical likelihood maximization

        Returns
        -------
        llr : float
            n times the log likelihood value for a given value of eta
        """
        return np.sum((self.endog - self.mu0) / (1.0 + eta * (self.endog - self.mu0)))

    def _ci_limits_mu(self, mu):
        """
        Calculates the difference between the log likelihood of mu_test and a
        specified critical value.

        Parameters
        ----------
        mu : float
           Hypothesized value of the mean.

        Returns
        -------
        diff : float
            The difference between the log likelihood value of mu0 and
            a specified value.
        """
        return self.test_mean(mu)[0] - self.r0

    def _find_gamma(self, gamma):
        """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        Used for confidence intervals for the mean

        Parameters
        ----------
        gamma : float
            Lagrange multiplier when computing confidence interval

        Returns
        -------
        diff : float
            The difference between the log-liklihood when the Lagrange
            multiplier is gamma and a pre-specified value
        """
        denom = np.sum((self.endog - gamma) ** (-1))
        new_weights = (self.endog - gamma) ** (-1) / denom
        return -2 * np.sum(np.log(self.nobs * new_weights)) - self.r0

    def _opt_var(self, nuisance_mu, pval=False):
        """
        This is the function to be optimized over a nuisance mean parameter
        to determine the likelihood ratio for the variance

        Parameters
        ----------
        nuisance_mu : float
            Value of a nuisance mean parameter

        Returns
        -------
        llr : float
            Log likelihood of a pre-specified variance holding the nuisance
            parameter constant
        """
        endog = self.endog
        nobs = self.nobs
        sig_data = (endog - nuisance_mu) ** 2 - self.sig2_0
        mu_data = endog - nuisance_mu
        est_vect = np.column_stack((mu_data, sig_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        if pval:
            return chi2.sf(-2 * llr, 1)
        return -2 * llr

    def _ci_limits_var(self, var):
        """
        Used to determine the confidence intervals for the variance.
        It calls test_var and when called by an optimizer,
        finds the value of sig2_0 that is chi2.ppf(significance-level)

        Parameters
        ----------
        var_test : float
            Hypothesized value of the variance

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at var_test and a
            pre-specified value.
        """
        return self.test_var(var)[0] - self.r0

    def _opt_skew(self, nuis_params):
        """
        Called by test_skew.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a  nuisance mean and variance parameter

        Returns
        -------
        llr : float
            The log likelihood ratio of a pre-specified skewness holding
            the nuisance parameters constant.
        """
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
        skew_data = (endog - nuis_params[0]) ** 3 / nuis_params[1] ** 1.5 - self.skew0
        est_vect = np.column_stack((mu_data, sig_data, skew_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1.0 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _opt_kurt(self, nuis_params):
        """
        Called by test_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        -------
        llr : float
            The log likelihood ratio of a pre-speified kurtosis holding the
            nuisance parameters constant
        """
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
        kurt_data = (endog - nuis_params[0]) ** 4 / nuis_params[1] ** 2 - 3 - self.kurt0
        est_vect = np.column_stack((mu_data, sig_data, kurt_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _opt_skew_kurt(self, nuis_params):
        """
        Called by test_joint_skew_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------
        nuis_params : 1darray
            An array with a nuisance mean and variance parameter

        Returns
        ------
        llr : float
            The log likelihood ratio of a pre-speified skewness and
            kurtosis holding the nuisance parameters constant.
        """
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
        skew_data = (endog - nuis_params[0]) ** 3 / nuis_params[1] ** 1.5 - self.skew0
        kurt_data = (endog - nuis_params[0]) ** 4 / nuis_params[1] ** 2 - 3 - self.kurt0
        est_vect = np.column_stack((mu_data, sig_data, skew_data, kurt_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1.0 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_skew(self, skew):
        """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of skewness

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at skew and a
            pre-specified value.
        """
        return self.test_skew(skew)[0] - self.r0

    def _ci_limits_kurt(self, kurt):
        """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of kurtosis

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at kurt and a
            pre-specified value.
        """
        return self.test_kurt(kurt)[0] - self.r0

    def _opt_correl(self, nuis_params, corr0, endog, nobs, x0, weights0):
        """
        Parameters
        ----------
        nuis_params : 1darray
            Array containing two nuisance means and two nuisance variances

        Returns
        -------
        llr : float
            The log-likelihood of the correlation coefficient holding nuisance
            parameters constant
        """
        mu1_data, mu2_data = (endog - nuis_params[::2]).T
        sig1_data = mu1_data ** 2 - nuis_params[1]
        sig2_data = mu2_data ** 2 - nuis_params[3]
        correl_data = mu1_data * mu2_data - corr0 * (nuis_params[1] * nuis_params[3]) ** 0.5
        est_vect = np.column_stack((mu1_data, sig1_data, mu2_data, sig2_data, correl_data))
        eta_star = self._modif_newton(x0, est_vect, weights0)
        denom = 1.0 + np.dot(est_vect, eta_star)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_corr(self, corr):
        return self.test_corr(corr)[0] - self.r0