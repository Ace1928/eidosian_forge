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
@unittest.skipUnless('GLOP' in INSTALLED_SOLVERS, 'GLOP is not installed.')
class TestGLOP(unittest.TestCase):

    def test_glop_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='GLOP')

    def test_glop_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='GLOP')

    def test_glop_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='GLOP')

    def test_glop_lp_3_no_preprocessing(self) -> None:
        from ortools.glop import parameters_pb2
        params = parameters_pb2.GlopParameters()
        params.use_preprocessing = False
        StandardTestLPs.test_lp_3(solver='GLOP', parameters_proto=params)

    @unittest.skip('Known limitation of the GLOP interface.')
    def test_glop_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='GLOP')

    def test_glop_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='GLOP')

    def test_glop_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='GLOP')

    def test_glop_lp_6_no_preprocessing(self) -> None:
        from ortools.glop import parameters_pb2
        params = parameters_pb2.GlopParameters()
        params.use_preprocessing = False
        StandardTestLPs.test_lp_6(solver='GLOP', parameters_proto=params)

    @unittest.skip('Known limitation of the GLOP interface.')
    def test_glop_lp_6(self) -> None:
        StandardTestLPs.test_lp_6(solver='GLOP')

    def test_glop_bad_parameters(self) -> None:
        x = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(x), [x <= 1])
        with self.assertRaises(cp.error.SolverError):
            prob.solve(solver='GLOP', parameters_proto='not a proto')

    def test_glop_time_limit(self) -> None:
        sth = sths.lp_1()
        sth.solve(solver='GLOP', time_limit_sec=1.0)