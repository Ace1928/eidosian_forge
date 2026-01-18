import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (

        Set default bandwiths based on domain values.

        Parameters
        ----------
        loc : array_like
            Values from the domain to which the kernel will
            be applied.
        bwm : scalar, optional
            A non-negative scalar that is used to multiply
            the default bandwidth.
        