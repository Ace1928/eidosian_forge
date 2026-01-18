from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class GenericSmoothers(AdditiveGamSmoother):
    """generic class for additive smooth components for GAM
    """

    def __init__(self, x, smoothers):
        self.smoothers = smoothers
        super().__init__(x, variable_names=None)

    def _make_smoothers_list(self):
        return self.smoothers