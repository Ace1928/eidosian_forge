import numpy as np
from statsmodels.emplike.descriptive import _OptFuncts

        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest.

        Parameters
        ----------
        nuisance_params: 1darray
            Parameters to be optimized over.

        Returns
        -------
        llr : float
            -2 x the log-likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        