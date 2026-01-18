from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _make_smoothers_list(self):
    smoothers = []
    for v in range(self.k_variables):
        uv_smoother = UnivariateCubicCyclicSplines(self.x[:, v], df=self.dfs[v], constraints=self.constraints, variable_name=self.variable_names[v])
        smoothers.append(uv_smoother)
    return smoothers