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
class TestECOS_BB(unittest.TestCase):

    def test_ecos_bb_explicit_only(self) -> None:
        """Test that ECOS_BB isn't chosen by default.
        """
        x = cp.Variable(1, name='x', integer=True)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [x >= 0])
        if INSTALLED_MI_SOLVERS != [cp.ECOS_BB]:
            prob.solve()
            assert prob.solver_stats.solver_name != cp.ECOS_BB
        else:
            with pytest.raises(cp.error.SolverError, match='You need a mixed-integer solver for this model'):
                prob.solve()

    def test_ecos_bb_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='ECOS_BB')

    def test_ecos_bb_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='ECOS_BB')
        StandardTestLPs.test_lp_1(solver='ECOS_BB')

    def test_ecos_bb_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='ECOS_BB')

    def test_ecos_bb_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='ECOS_BB')

    def test_ecos_bb_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='ECOS_BB')

    def test_ecos_bb_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='ECOS_BB')

    def test_ecos_bb_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='ECOS_BB')

    def test_ecos_bb_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='ECOS_BB')

    def test_ecos_bb_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='ECOS_BB')

    def test_ecos_bb_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='ECOS_BB')
        StandardTestSOCPs.test_socp_3ax1(solver='ECOS_BB')

    def test_ecos_bb_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='ECOS_BB')

    def test_ecos_bb_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='ECOS_BB')

    def test_ecos_bb_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='ECOS_BB')

    @pytest.mark.skip(reason='Known bug in ECOS BB')
    def test_ecos_bb_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='ECOS_BB')

    def test_ecos_bb_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='ECOS_BB')

    def test_ecos_bb_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='ECOS_BB')

    @pytest.mark.skip(reason='Known bug in ECOS BB')
    def test_ecos_bb_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='ECOS_BB')