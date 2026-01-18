import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings
def _iter__(self):
    """initialize iteration ?, should be removed

        """
    self.iter = 0
    self.dev = np.inf
    return self