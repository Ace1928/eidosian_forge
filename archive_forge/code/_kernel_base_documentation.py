import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels

        Returns the cross-validation least squares bandwidth parameter(s).

        Notes
        -----
        For more details see pp. 16, 27 in Ref. [1] (see module docstring).

        Returns the value of the bandwidth that maximizes the integrated mean
        square error between the estimated and actual distribution.  The
        integrated mean square error (IMSE) is given by:

        .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

        This is the general formula for the IMSE.  The IMSE differs for
        conditional (``KDEMultivariateConditional``) and unconditional
        (``KDEMultivariate``) kernel density estimation.
        