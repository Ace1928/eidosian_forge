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
@unittest.skipUnless('NAG' in INSTALLED_SOLVERS, 'NAG is not installed.')
class TestNAG(unittest.TestCase):

    def test_nag_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='NAG')

    def test_nag_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='NAG')

    def test_nag_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='NAG')

    def test_nag_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='NAG')

    def test_nag_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='NAG')

    def test_nag_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='NAG')

    def test_nag_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='NAG')

    def test_nag_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='NAG')

    def test_nag_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='NAG')

    def test_nag_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='NAG')
        StandardTestSOCPs.test_socp_3ax1(solver='NAG')