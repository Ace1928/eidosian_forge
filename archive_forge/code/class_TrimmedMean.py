import numpy as np
class TrimmedMean(RobustNorm):
    """
    Trimmed mean function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        2.0.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, c=2.0):
        self.c = c

    def _subset(self, z):
        """
        Least trimmed mean is defined piecewise over the range of z.
        """
        z = np.asarray(z)
        return np.less_equal(np.abs(z), self.c)

    def rho(self, z):
        """
        The robust criterion function for least trimmed mean.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = (1/2.)*z**2    for \\|z\\| <= c

            rho(z) = (1/2.)*c**2              for \\|z\\| > c
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z ** 2 * 0.5 + (1 - test) * self.c ** 2 * 0.5

    def psi(self, z):
        """
        The psi function for least trimmed mean

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z              for \\|z\\| <= c

            psi(z) = 0              for \\|z\\| > c
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z

    def weights(self, z):
        """
        Least trimmed mean weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1             for \\|z\\| <= c

            weights(z) = 0             for \\|z\\| > c
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test

    def psi_deriv(self, z):
        """
        The derivative of least trimmed mean psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        test = self._subset(z)
        return test