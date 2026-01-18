import numpy as np
class VarianceFunction:
    """
    Relates the variance of a random variable to its mean. Defaults to 1.

    Methods
    -------
    call
        Returns an array of ones that is the same shape as `mu`

    Notes
    -----
    After a variance function is initialized, its call method can be used.

    Alias for VarianceFunction:
    constant = VarianceFunction()

    See Also
    --------
    statsmodels.genmod.families.family
    """

    def __call__(self, mu):
        """
        Default variance function

        Parameters
        ----------
        mu : array_like
            mean parameters

        Returns
        -------
        v : ndarray
            ones(mu.shape)
        """
        mu = np.asarray(mu)
        return np.ones(mu.shape, np.float64)

    def deriv(self, mu):
        """
        Derivative of the variance function v'(mu)
        """
        return np.zeros_like(mu)