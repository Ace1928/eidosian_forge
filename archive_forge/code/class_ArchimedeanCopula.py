import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
class ArchimedeanCopula(Copula):
    """Base class for Archimedean copulas

    Parameters
    ----------
    transform : instance of transformation class
        Archimedean generator with required methods including first and second
        derivatives
    args : tuple
        Optional copula parameters. Copula parameters can be either provided
        when creating the instance or as arguments when calling methods.
    k_dim : int
        Dimension, number of components in the multivariate random variable.
        Currently only bivariate copulas are verified. Support for more than
        2 dimension is incomplete.
    """

    def __init__(self, transform, args=(), k_dim=2):
        super().__init__(k_dim=k_dim)
        self.args = args
        self.transform = transform
        self.k_args = 1

    def _handle_args(self, args):
        if isinstance(args, np.ndarray):
            args = tuple(args)
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) == 0 or args == (None,):
            args = self.args
        return args

    def _handle_u(self, u):
        u = np.asarray(u)
        if u.shape[-1] != self.k_dim:
            import warnings
            warnings.warn('u has different dimension than k_dim. This will raise exception in future versions', FutureWarning)
        return u

    def cdf(self, u, args=()):
        """Evaluate cdf of Archimedean copula."""
        args = self._handle_args(args)
        u = self._handle_u(u)
        axis = -1
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
        cdfv = phi_inv(phi(u, *args).sum(axis), *args)
        out = cdfv if isinstance(cdfv, np.ndarray) else None
        cdfv = np.clip(cdfv, 0.0, 1.0, out=out)
        return cdfv

    def pdf(self, u, args=()):
        """Evaluate pdf of Archimedean copula."""
        u = self._handle_u(u)
        args = self._handle_args(args)
        axis = -1
        phi_d1 = self.transform.deriv
        if u.shape[-1] == 2:
            psi_d = self.transform.deriv2_inverse
        elif u.shape[-1] == 3:
            psi_d = self.transform.deriv3_inverse
        elif u.shape[-1] == 4:
            psi_d = self.transform.deriv4_inverse
        else:
            k = u.shape[-1]

            def psi_d(*args):
                return self.transform.derivk_inverse(k, *args)
        psi = self.transform.evaluate(u, *args).sum(axis)
        pdfv = np.prod(phi_d1(u, *args), axis)
        pdfv *= psi_d(psi, *args)
        return np.abs(pdfv)

    def logpdf(self, u, args=()):
        """Evaluate log pdf of multivariate Archimedean copula."""
        u = self._handle_u(u)
        args = self._handle_args(args)
        axis = -1
        phi_d1 = self.transform.deriv
        if u.shape[-1] == 2:
            psi_d = self.transform.deriv2_inverse
        elif u.shape[-1] == 3:
            psi_d = self.transform.deriv3_inverse
        elif u.shape[-1] == 4:
            psi_d = self.transform.deriv4_inverse
        else:
            k = u.shape[-1]

            def psi_d(*args):
                return self.transform.derivk_inverse(k, *args)
        psi = self.transform.evaluate(u, *args).sum(axis)
        logpdfv = np.sum(np.log(np.abs(phi_d1(u, *args))), axis)
        logpdfv += np.log(np.abs(psi_d(psi, *args)))
        return logpdfv

    def _arg_from_tau(self, tau):
        return self.theta_from_tau(tau)