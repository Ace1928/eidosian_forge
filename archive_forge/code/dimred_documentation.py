import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning

        Fit the covariance reduction model.

        Parameters
        ----------
        start_params : array_like
            Starting value for the projection matrix. May be
            rectangular, or flattened.
        maxiter : int
            The maximum number of gradient steps to take.
        gtol : float
            Convergence criterion for the gradient norm.

        Returns
        -------
        A results instance that can be used to access the
        fitted parameters.
        