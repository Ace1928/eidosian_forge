from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class UnivariatePolynomialSmoother(UnivariateGamSmoother):
    """polynomial single smooth component
    """

    def __init__(self, x, degree, variable_name='x'):
        self.degree = degree
        super().__init__(x, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        """
        given a vector x returns poly=(1, x, x^2, ..., x^degree)
        and its first and second derivative
        """
        basis = np.zeros(shape=(self.nobs, self.degree))
        der_basis = np.zeros(shape=(self.nobs, self.degree))
        der2_basis = np.zeros(shape=(self.nobs, self.degree))
        for i in range(self.degree):
            dg = i + 1
            basis[:, i] = self.x ** dg
            der_basis[:, i] = dg * self.x ** (dg - 1)
            if dg > 1:
                der2_basis[:, i] = dg * (dg - 1) * self.x ** (dg - 2)
            else:
                der2_basis[:, i] = 0
        cov_der2 = np.dot(der2_basis.T, der2_basis)
        return (basis, der_basis, der2_basis, cov_der2)