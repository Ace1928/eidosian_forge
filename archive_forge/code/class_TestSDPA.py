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
@unittest.skipUnless('SDPA' in INSTALLED_SOLVERS, 'SDPA is not installed.')
class TestSDPA(BaseTest):

    def test_sdpa_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='SDPA')

    def test_sdpa_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='SDPA')

    def test_sdpa_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='SDPA')

    def test_sdpa_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SDPA')

    def test_sdpa_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SDPA')

    def test_sdpa_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='SDPA', betaBar=0.1, gammaStar=0.8, epsilonDash=8e-06)

    def test_sdpa_lp_7(self) -> None:
        StandardTestLPs.test_lp_7(solver='SDPA')

    def test_sdpa_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='SDPA')

    def test_sdpa_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='SDPA')

    def test_sdpa_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='SDPA')

    def test_sdpa_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='SDPA')
        StandardTestSOCPs.test_socp_3ax1(solver='SDPA')

    def test_sdpa_sdp_1(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='SDPA')
        StandardTestSDPs.test_sdp_1max(solver='SDPA')

    def test_sdpa_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='SDPA')