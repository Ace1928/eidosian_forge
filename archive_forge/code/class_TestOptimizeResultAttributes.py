import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
class TestOptimizeResultAttributes:

    def setup_method(self):
        self.x0 = [5, 5]
        self.func = optimize.rosen
        self.jac = optimize.rosen_der
        self.hess = optimize.rosen_hess
        self.hessp = optimize.rosen_hess_prod
        self.bounds = [(0.0, 10.0), (0.0, 10.0)]

    def test_attributes_present(self):
        attributes = ['nit', 'nfev', 'x', 'success', 'status', 'fun', 'message']
        skip = {'cobyla': ['nit']}
        for method in MINIMIZE_METHODS:
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'Method .+ does not use (gradient|Hessian.*) information')
                res = optimize.minimize(self.func, self.x0, method=method, jac=self.jac, hess=self.hess, hessp=self.hessp)
            for attribute in attributes:
                if method in skip and attribute in skip[method]:
                    continue
                assert hasattr(res, attribute)
                assert attribute in dir(res)
            assert isinstance(res.message, str)