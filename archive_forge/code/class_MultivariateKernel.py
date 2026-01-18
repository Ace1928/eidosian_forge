import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
class MultivariateKernel:
    """
    Base class for multivariate kernels.

    An instance of MultivariateKernel implements a `call` method having
    signature `call(x, loc)`, returning the kernel weights comparing `x`
    (a 1d ndarray) to each row of `loc` (a 2d ndarray).
    """

    def call(self, x, loc):
        raise NotImplementedError

    def set_bandwidth(self, bw):
        """
        Set the bandwidth to the given vector.

        Parameters
        ----------
        bw : array_like
            A vector of non-negative bandwidth values.
        """
        self.bw = bw
        self._setup()

    def _setup(self):
        self.bwk = np.prod(self.bw)
        self.bw2 = self.bw * self.bw

    def set_default_bw(self, loc, bwm=None):
        """
        Set default bandwiths based on domain values.

        Parameters
        ----------
        loc : array_like
            Values from the domain to which the kernel will
            be applied.
        bwm : scalar, optional
            A non-negative scalar that is used to multiply
            the default bandwidth.
        """
        sd = loc.std(0)
        q25, q75 = np.percentile(loc, [25, 75], axis=0)
        iqr = (q75 - q25) / 1.349
        bw = np.where(iqr < sd, iqr, sd)
        bw *= 0.9 / loc.shape[0] ** 0.2
        if bwm is not None:
            bw *= bwm
        self.bw = np.asarray(bw, dtype=np.float64)
        self._setup()