from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def _check_alpha(self, alpha):
    """check and convert alpha to required list format

        Parameters
        ----------
        alpha : scalar, list or array_like
            penalization weight

        Returns
        -------
        alpha : list
            penalization weight, list with length equal to the number of
            smooth terms
        """
    if not isinstance(alpha, Iterable):
        alpha = [alpha] * len(self.smoother.smoothers)
    elif not isinstance(alpha, list):
        alpha = list(alpha)
    return alpha