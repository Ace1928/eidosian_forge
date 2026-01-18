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
@pytest.mark.skipif('CBC' not in INSTALLED_SOLVERS, reason='CBC is not installed.')
class TestCBC:

    def _cylp_checks_isProvenInfeasible():
        try:
            from cylp.cy.CyCbcModel import problemStatus
            return problemStatus[0] == 'search completed'
        except ImportError:
            return False

    def test_cbc_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='CBC', duals=False)

    def test_cbc_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CBC', duals=False)

    def test_cbc_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CBC', duals=False)

    def test_cbc_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='CBC')

    def test_cbc_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CBC')

    def test_cbc_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='CBC', duals=False)

    def test_cbc_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='CBC')

    def test_cbc_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='CBC')

    def test_cbc_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='CBC')

    def test_cbc_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='CBC')

    @pytest.mark.skipif(not _cylp_checks_isProvenInfeasible(), reason='CyLP <= 0.91.4 has no working integer infeasibility detection')
    def test_cbc_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='CBC')

    @pytest.mark.parametrize('opts', [pytest.param(opts, id=next(iter(opts.keys()))) for opts in [{'dualTolerance': 1.0}, {'primalTolerance': 1.0}, {'maxNumIteration': 1}, {'scaling': 0}, {'optimizationDirection': 'max'}, {'presolve': 'off'}]])
    def test_cbc_lp_options(self, opts: dict, capfd: pytest.LogCaptureFixture) -> None:
        """
        Validate that cylp is actually using each option.

        Tentative approach: run model with verbose output with or without the specified
        option; verbose output should be different each way.
        """
        fflush()
        capfd.readouterr()
        sth = sths.lp_4()
        sth.solve(solver='CBC', logLevel=2)
        fflush()
        base = capfd.readouterr()
        try:
            sth.solve(solver='CBC', logLevel=2, **opts)
        except Exception:
            pass
        else:
            fflush()
            with_opt = capfd.readouterr()
            assert base != with_opt

    def test_cbc_lp_logging(self, capfd: pytest.LogCaptureFixture) -> None:
        """Validate that logLevel parameter is passed to solver"""
        fflush()
        capfd.readouterr()
        StandardTestLPs.test_lp_0(solver='CBC', duals=False, logLevel=0)
        fflush()
        quiet_output = capfd.readouterr()
        StandardTestLPs.test_lp_0(solver='CBC', duals=False, logLevel=5)
        fflush()
        verbose_output = capfd.readouterr()
        assert len(verbose_output.out) > len(quiet_output.out)
        StandardTestLPs.test_mi_lp_0(solver='CBC', logLevel=0)
        fflush()
        quiet_output = capfd.readouterr()
        StandardTestLPs.test_mi_lp_0(solver='CBC', logLevel=5)
        fflush()
        verbose_output = capfd.readouterr()
        assert len(verbose_output.out) > len(quiet_output.out)