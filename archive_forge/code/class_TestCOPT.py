import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
@unittest.skipUnless('COPT' in INSTALLED_SOLVERS, 'COPT is not installed.')
class TestCOPT(unittest.TestCase):

    def test_copt_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='COPT')

    def test_copt_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='COPT')

    def test_copt_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='COPT')

    def test_copt_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='COPT')

    def test_copt_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='COPT')

    def test_copt_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='COPT')

    def test_copt_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='COPT')

    def test_copt_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='COPT', places=3)

    def test_copt_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='COPT')

    def test_copt_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='COPT')
        StandardTestSOCPs.test_socp_3ax1(solver='COPT')

    def test_copt_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='COPT')

    def test_copt_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='COPT')

    def test_copt_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='COPT')

    def test_copt_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='COPT')

    def test_copt_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='COPT')

    def test_copt_mi_socp_1(self) -> None:
        with pytest.raises(cp.error.SolverError, match='do not support'):
            StandardTestSOCPs.test_mi_socp_1(solver='COPT')

    def test_copt_sdp_1min(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='COPT')

    def test_copt_sdp_1max(self) -> None:
        StandardTestSDPs.test_sdp_1max(solver='COPT')

    def test_copt_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='COPT', places=2)

    def test_copt_params(self) -> None:
        n = 10
        m = 4
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)
        with self.assertRaises(AttributeError):
            problem.solve(solver=cp.COPT, invalid_kwarg=None)
        problem.solve(solver=cp.COPT, feastol=1e-09)