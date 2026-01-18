import numpy as np
class LeastSquares(RobustNorm):
    """
    Least squares rho for M-estimation and its derived functions.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def rho(self, z):
        """
        The least squares estimator rho function

        Parameters
        ----------
        z : ndarray
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = (1/2.)*z**2
        """
        return z ** 2 * 0.5

    def psi(self, z):
        """
        The psi function for the least squares estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z
        """
        return np.asarray(z)

    def weights(self, z):
        """
        The least squares estimator weighting function for the IRLS algorithm.

        The psi function scaled by the input z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = np.ones(z.shape)
        """
        z = np.asarray(z)
        return np.ones(z.shape, np.float64)

    def psi_deriv(self, z):
        """
        The derivative of the least squares psi function.

        Returns
        -------
        psi_deriv : ndarray
            ones(z.shape)

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.ones(z.shape, np.float64)