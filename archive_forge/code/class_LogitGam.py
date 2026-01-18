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
class LogitGam(PenalizedMixin, Logit):
    """Generalized Additive model for discrete Logit

    This subclasses discrete_model Logit.

    Warning: not all inherited methods might take correctly account of the
    penalization

    not verified yet.
    """

    def __init__(self, endog, smoother, alpha, *args, **kwargs):
        if not isinstance(alpha, Iterable):
            alpha = np.array([alpha] * len(smoother.smoothers))
        self.smoother = smoother
        self.alpha = alpha
        self.pen_weight = 1
        penal = MultivariateGamPenalty(smoother, alpha=alpha)
        super().__init__(endog, smoother.basis, *args, penal=penal, **kwargs)