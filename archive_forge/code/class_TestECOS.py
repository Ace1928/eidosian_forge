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
class TestECOS(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')
        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_ecos_options(self) -> None:
        """Test that all the ECOS solver options work.
        """
        EPS = 0.0001
        prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
        for i in range(2):
            prob.solve(solver=cp.ECOS, feastol=EPS, abstol=EPS, reltol=EPS, feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS, max_iters=20, verbose=True, warm_start=True)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_ecos_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='ECOS')

    def test_ecos_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='ECOS')

    def test_ecos_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='ECOS')

    def test_ecos_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='ECOS')

    def test_ecos_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='ECOS')

    def test_ecos_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='ECOS')

    def test_ecos_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='ECOS')

    def test_ecos_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='ECOS')

    def test_ecos_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='ECOS')

    def test_ecos_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='ECOS')
        StandardTestSOCPs.test_socp_3ax1(solver='ECOS')

    def test_ecos_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='ECOS')

    def test_ecos_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='ECOS')