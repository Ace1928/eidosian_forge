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
class TestSCIPY(unittest.TestCase):

    def setUp(self):
        import scipy
        self.d = Version(scipy.__version__) >= Version('1.7.0')

    def test_scipy_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='SCIPY', duals=self.d)

    def test_scipy_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='SCIPY', duals=self.d)

    def test_scipy_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='SCIPY', duals=self.d)

    def test_scipy_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SCIPY')

    def test_scipy_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SCIPY')

    def test_scipy_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='SCIPY', duals=self.d)

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_4(self) -> None:
        StandardTestLPs.test_mi_lp_4(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_time_limit_reached(self) -> None:
        sth = sths.mi_lp_7()
        sth.solve(solver='SCIPY', scipy_options={'time_limit': 100.0})