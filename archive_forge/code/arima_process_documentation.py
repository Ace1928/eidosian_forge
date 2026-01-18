from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like

        Make MA polynomial invertible by inverting roots inside unit circle.

        Parameters
        ----------
        retnew : bool
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial.

        Returns
        -------
        manew : ndarray
           A new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : bool
           True if the MA lag-polynomial was already invertible, returned if
           retnew is false.
        armaprocess : new instance of class
           If retnew is true, then return a new instance with invertible
           MA-polynomial.
        