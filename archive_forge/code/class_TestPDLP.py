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
@unittest.skipUnless('PDLP' in INSTALLED_SOLVERS, 'PDLP is not installed.')
class TestPDLP(unittest.TestCase):

    def test_pdlp_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='PDLP')

    def test_pdlp_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='PDLP')

    def test_pdlp_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='PDLP')

    def test_pdlp_lp_3(self) -> None:
        sth = sths.lp_3()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='PDLP')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)

    def test_pdlp_lp_3_no_presolve(self) -> None:
        from ortools.pdlp import solvers_pb2
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.presolve_options.use_glop = False
        StandardTestLPs.test_lp_3(solver='PDLP', parameters_proto=params)

    def test_pdlp_lp_4(self) -> None:
        sth = sths.lp_4()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='PDLP')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)

    def test_pdlp_lp_4_no_presolve(self) -> None:
        from ortools.pdlp import solvers_pb2
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.presolve_options.use_glop = False
        StandardTestLPs.test_lp_4(solver='PDLP', parameters_proto=params)

    def test_pdlp_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='PDLP')

    def test_pdlp_lp_6(self) -> None:
        sth = sths.lp_6()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='PDLP')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)

    def test_pdlp_lp_6_no_presolve(self) -> None:
        from ortools.pdlp import solvers_pb2
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.presolve_options.use_glop = False
        StandardTestLPs.test_lp_6(solver='PDLP', parameters_proto=params)

    def test_pdlp_bad_parameters(self) -> None:
        x = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(x), [x <= 1])
        with self.assertRaises(cp.error.SolverError):
            prob.solve(solver='PDLP', parameters_proto='not a proto')

    def test_pdlp_time_limit(self) -> None:
        sth = sths.lp_1()
        sth.solve(solver='PDLP', time_limit_sec=1.0)