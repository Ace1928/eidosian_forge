import numpy as np
import scipy.stats
import warnings
class Probit(CDFLink):
    """
    The probit (standard normal CDF) transform

    Notes
    -----
    g(p) = scipy.stats.norm.ppf(p)

    probit is an alias of CDFLink.
    """

    def inverse_deriv2(self, z):
        """
        Second derivative of the inverse link function

        This is the derivative of the pdf in a CDFLink

        """
        return -z * self.dbn.pdf(z)

    def deriv2(self, p):
        """
        Second derivative of the link function g''(p)

        """
        p = self._clean(p)
        linpred = self.dbn.ppf(p)
        return linpred / self.dbn.pdf(linpred) ** 2