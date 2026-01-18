import numpy as np
import scipy.stats
import warnings
class LogLog(Logit):
    """
    The log-log transform

    LogLog inherits from Logit in order to have access to its _clean method
    for the link and its derivative.
    """

    def __call__(self, p):
        """
        Log-Log transform link function

        Parameters
        ----------
        p : ndarray
            Mean parameters

        Returns
        -------
        z : ndarray
            The LogLog transform of `p`

        Notes
        -----
        g(p) = -log(-log(p))
        """
        p = self._clean(p)
        return -np.log(-np.log(p))

    def inverse(self, z):
        """
        Inverse of Log-Log transform link function


        Parameters
        ----------
        z : array_like
            The value of the inverse of the LogLog link function at `p`

        Returns
        -------
        p : ndarray
            Mean parameters

        Notes
        -----
        g^(-1)(`z`) = exp(-exp(-`z`))
        """
        return np.exp(-np.exp(-z))

    def deriv(self, p):
        """
        Derivative of Log-Log transform link function

        Parameters
        ----------
        p : array_like
            Mean parameters

        Returns
        -------
        g'(p) : ndarray
            The derivative of the LogLog transform link function

        Notes
        -----
        g'(p) = - 1 /(p * log(p))
        """
        p = self._clean(p)
        return -1.0 / (p * np.log(p))

    def deriv2(self, p):
        """
        Second derivative of the Log-Log link function

        Parameters
        ----------
        p : array_like
            Mean parameters

        Returns
        -------
        g''(p) : ndarray
            The second derivative of the LogLog link function
        """
        p = self._clean(p)
        d2 = (1 + np.log(p)) / (p * np.log(p)) ** 2
        return d2

    def inverse_deriv(self, z):
        """
        Derivative of the inverse of the Log-Log transform link function

        Parameters
        ----------
        z : array_like
            The value of the inverse of the LogLog link function at `p`

        Returns
        -------
        g^(-1)'(z) : ndarray
            The derivative of the inverse of the LogLog link function
        """
        return np.exp(-np.exp(-z) - z)

    def inverse_deriv2(self, z):
        """
        Second derivative of the inverse of the Log-Log transform link function

        Parameters
        ----------
        z : array_like
            The value of the inverse of the LogLog link function at `p`

        Returns
        -------
        g^(-1)''(z) : ndarray
            The second derivative of the inverse of the LogLog link function
        """
        return self.inverse_deriv(z) * (np.exp(-z) - 1)