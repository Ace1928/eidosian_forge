from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class UnivariateGamSmoother(with_metaclass(ABCMeta)):
    """Base Class for single smooth component
    """

    def __init__(self, x, constraints=None, variable_name='x'):
        self.x = x
        self.constraints = constraints
        self.variable_name = variable_name
        self.nobs, self.k_variables = (len(x), 1)
        base4 = self._smooth_basis_for_single_variable()
        if constraints == 'center':
            constraints = base4[0].mean(0)[None, :]
        if constraints is not None and (not isinstance(constraints, str)):
            ctransf = transf_constraints(constraints)
            self.ctransf = ctransf
        elif not hasattr(self, 'ctransf'):
            self.ctransf = None
        self.basis, self.der_basis, self.der2_basis, self.cov_der2 = base4
        if self.ctransf is not None:
            ctransf = self.ctransf
            if base4[0] is not None:
                self.basis = base4[0].dot(ctransf)
            if base4[1] is not None:
                self.der_basis = base4[1].dot(ctransf)
            if base4[2] is not None:
                self.der2_basis = base4[2].dot(ctransf)
            if base4[3] is not None:
                self.cov_der2 = ctransf.T.dot(base4[3]).dot(ctransf)
        self.dim_basis = self.basis.shape[1]
        self.col_names = [self.variable_name + '_s' + str(i) for i in range(self.dim_basis)]

    @abstractmethod
    def _smooth_basis_for_single_variable(self):
        return