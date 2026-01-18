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
@unittest.skipUnless('SCIP' in INSTALLED_SOLVERS, 'SCIP is not installed.')
class TestSCIP(unittest.TestCase):

    def test_scip_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='SCIP')

    def test_scip_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='SCIP')

    def test_scip_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='SCIP', duals=False)

    def test_scip_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SCIP')

    def test_scip_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SCIP')

    def test_scip_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='SCIP')

    def test_scip_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='SCIP', places=2, duals=False)

    def test_scip_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='SCIP', places=2, duals=False)

    def test_scip_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='SCIP', duals=False)
        StandardTestSOCPs.test_socp_3ax1(solver='SCIP', duals=False)

    def test_scip_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='SCIP')

    def test_scip_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='SCIP')

    def test_scip_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='SCIP')

    def test_scip_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='SCIP')

    def test_scip_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='SCIP')

    def test_scip_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='SCIP', places=3)

    def test_scip_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='SCIP')

    def get_simple_problem(self):
        """Example problem that can be used within additional tests."""
        x = cp.Variable()
        y = cp.Variable()
        constraints = [x >= 0, y >= 1, x + y <= 4]
        obj = cp.Maximize(x)
        prob = cp.Problem(obj, constraints)
        return prob

    def test_scip_test_params__no_params_set(self) -> None:
        prob = self.get_simple_problem()
        prob.solve(solver='SCIP')
        assert prob.value == 3

    def test_scip_test_params__valid_params(self) -> None:
        prob = self.get_simple_problem()
        prob.solve(solver='SCIP', gp=False)
        assert prob.value == 3

    def test_scip_test_params__valid_scip_params(self) -> None:
        prob = self.get_simple_problem()
        prob.solve(solver='SCIP', scip_params={'lp/fastmip': 1, 'limits/gap': 0.1})
        assert prob.value == 3

    def test_scip_test_params__invalid_params(self) -> None:
        prob = self.get_simple_problem()
        with pytest.raises(KeyError) as ke:
            prob.solve(solver='SCIP', a='what?')
            exc = "One or more solver params in ['a'] are not valid: 'Not a valid parameter name'"
            assert ke.exception == exc

    def test_scip_test_params__invalid_scip_params(self) -> None:
        prob = self.get_simple_problem()
        with pytest.raises(KeyError) as ke:
            prob.solve(solver='SCIP', scip_params={'a': 'what?'})
            exc = "One or more scip params in ['a'] are not valid: 'Not a valid parameter name'"
            assert ke.exception == exc

    def test_scip_time_limit_reached(self) -> None:
        sth = sths.mi_lp_7()
        with pytest.raises(cp.error.SolverError) as se:
            sth.solve(solver='SCIP', scip_params={'limits/time': 0.0})
            exc = "Solver 'SCIP' failed. Try another solver, or solve with verbose=True for more information."
            assert str(se.value) == exc