from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class UnivariateGenericSmoother(UnivariateGamSmoother):
    """Generic single smooth component
    """

    def __init__(self, x, basis, der_basis, der2_basis, cov_der2, variable_name='x'):
        self.basis = basis
        self.der_basis = der_basis
        self.der2_basis = der2_basis
        self.cov_der2 = cov_der2
        super().__init__(x, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        return (self.basis, self.der_basis, self.der2_basis, self.cov_der2)