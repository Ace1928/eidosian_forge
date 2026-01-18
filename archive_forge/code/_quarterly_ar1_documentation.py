import warnings
import numpy as np
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tools import (

    AR(1) model for monthly growth rates aggregated to quarterly frequency

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`

    Notes
    -----
    This model is internal, used to estimate starting parameters for the
    DynamicFactorMQ class. The model is:

    .. math::

        y_t & = \begin{bmatrix} 1 & 2 & 3 & 2 & 1 \end{bmatrix} \alpha_t \\
        \alpha_t & = \begin{bmatrix}
            \phi & 0 & 0 & 0 & 0 \\
               1 & 0 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 & 0 \\
               0 & 0 & 1 & 0 & 0 \\
               0 & 0 & 0 & 1 & 0 \\
        \end{bmatrix} +
        \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} \varepsilon_t

    The two parameters to be estimated are :math:`\phi` and :math:`\sigma^2`.

    It supports fitting via the usual quasi-Newton methods, as well as using
    the EM algorithm.

    