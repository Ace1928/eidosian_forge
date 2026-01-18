import numpy as np
class RamsayE(RobustNorm):
    """
    Ramsay's Ea for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        0.3.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=0.3):
        self.a = a

    def rho(self, z):
        """
        The robust criterion function for Ramsay's Ea.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = a**-2 * (1 - exp(-a*\\|z\\|)*(1 + a*\\|z\\|))
        """
        z = np.asarray(z)
        return (1 - np.exp(-self.a * np.abs(z)) * (1 + self.a * np.abs(z))) / self.a ** 2

    def psi(self, z):
        """
        The psi function for Ramsay's Ea estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z*exp(-a*\\|z\\|)
        """
        z = np.asarray(z)
        return z * np.exp(-self.a * np.abs(z))

    def weights(self, z):
        """
        Ramsay's Ea weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = exp(-a*\\|z\\|)
        """
        z = np.asarray(z)
        return np.exp(-self.a * np.abs(z))

    def psi_deriv(self, z):
        """
        The derivative of Ramsay's Ea psi function.

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        a = self.a
        x = np.exp(-a * np.abs(z))
        dx = -a * x * np.sign(z)
        y = z
        dy = 1
        return x * dy + y * dx