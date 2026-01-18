import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class DimReductionResults(model.Results):
    """
    Results class for a dimension reduction regression.

    Notes
    -----
    The `params` attribute is a matrix whose columns span
    the effective dimension reduction (EDR) space.  Some
    methods produce a corresponding set of eigenvalues
    (`eigs`) that indicate how much information is contained
    in each basis direction.
    """

    def __init__(self, model, params, eigs):
        super().__init__(model, params)
        self.eigs = eigs