import unittest
import numpy as np
import pytest
import scipy as sp
import cvxpy as cp
from cvxpy import settings as s
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as INSTALLED_MI
from cvxpy.reductions.solvers.defines import MI_SOCP_SOLVERS as MI_SOCP
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
class TestSlacks(BaseTest):
    AFF_LP_CASES = [[a2d.NONNEG], []]
    AFF_SOCP_CASES = [[a2d.NONNEG, a2d.SOC], [a2d.NONNEG], [a2d.SOC], []]
    AFF_EXP_CASES = [[a2d.NONNEG, a2d.EXP], [a2d.NONNEG], [a2d.EXP], []]
    AFF_PCP_CASES = [[a2d.NONNEG], [a2d.POW3D], []]
    AFF_MIXED_CASES = [[a2d.NONNEG], []]

    @staticmethod
    def simulate_chain(in_prob, affine, **solve_kwargs):
        reductions = [Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()]
        chain = Chain(None, reductions)
        cone_prog, inv_prob2cone = chain.apply(in_prob)
        cone_prog = ConicSolver().format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
        data, inv_data = a2d.Slacks.apply(cone_prog, affine)
        G, h, f, K_dir, K_aff = (data[s.A], data[s.B], data[s.C], data['K_dir'], data['K_aff'])
        G = sp.sparse.csc_matrix(G)
        y = cp.Variable(shape=(G.shape[1],))
        objective = cp.Minimize(f @ y)
        aff_con = TestSlacks.set_affine_constraints(G, h, y, K_aff)
        dir_con = TestSlacks.set_direct_constraints(y, K_dir)
        int_con = TestSlacks.set_integer_constraints(y, data)
        constraints = aff_con + dir_con + int_con
        slack_prob = cp.Problem(objective, constraints)
        slack_prob.solve(**solve_kwargs)
        slack_prims = {a2d.FREE: y[:cone_prog.x.size].value}
        slack_sol = Solution(slack_prob.status, slack_prob.value, slack_prims, None, dict())
        cone_sol = a2d.Slacks.invert(slack_sol, inv_data)
        in_prob_sol = chain.invert(cone_sol, inv_prob2cone)
        in_prob.unpack(in_prob_sol)

    @staticmethod
    def set_affine_constraints(G, h, y, K_aff):
        constraints = []
        i = 0
        if K_aff[a2d.ZERO]:
            dim = K_aff[a2d.ZERO]
            constraints.append(G[i:i + dim, :] @ y == h[i:i + dim])
            i += dim
        if K_aff[a2d.NONNEG]:
            dim = K_aff[a2d.NONNEG]
            constraints.append(G[i:i + dim, :] @ y <= h[i:i + dim])
            i += dim
        for dim in K_aff[a2d.SOC]:
            expr = h[i:i + dim] - G[i:i + dim, :] @ y
            constraints.append(SOC(expr[0], expr[1:]))
            i += dim
        if K_aff[a2d.EXP]:
            dim = 3 * K_aff[a2d.EXP]
            expr = cp.reshape(h[i:i + dim] - G[i:i + dim, :] @ y, (dim // 3, 3), order='C')
            constraints.append(ExpCone(expr[:, 0], expr[:, 1], expr[:, 2]))
            i += dim
        if K_aff[a2d.POW3D]:
            alpha = np.array(K_aff[a2d.POW3D])
            expr = cp.reshape(h[i:] - G[i:, :] @ y, (alpha.size, 3), order='C')
            constraints.append(PowCone3D(expr[:, 0], expr[:, 1], expr[:, 2], alpha))
        return constraints

    @staticmethod
    def set_direct_constraints(y, K_dir):
        constraints = []
        i = K_dir[a2d.FREE]
        if K_dir[a2d.NONNEG]:
            dim = K_dir[a2d.NONNEG]
            constraints.append(y[i:i + dim] >= 0)
            i += dim
        for dim in K_dir[a2d.SOC]:
            constraints.append(SOC(y[i], y[i + 1:i + dim]))
            i += dim
        if K_dir[a2d.EXP]:
            dim = 3 * K_dir[a2d.EXP]
            expr = cp.reshape(y[i:i + dim], (dim // 3, 3), order='C')
            constraints.append(ExpCone(expr[:, 0], expr[:, 1], expr[:, 2]))
            i += dim
        if K_dir[a2d.POW3D]:
            alpha = np.array(K_dir[a2d.POW3D])
            expr = cp.reshape(y[i:], (alpha.size, 3), order='C')
            constraints.append(PowCone3D(expr[:, 0], expr[:, 1], expr[:, 2], alpha))
        return constraints

    @staticmethod
    def set_integer_constraints(y, data):
        constraints = []
        if data[s.BOOL_IDX]:
            expr = y[data[s.BOOL_IDX]]
            z = cp.Variable(shape=(expr.size,), boolean=True)
            constraints.append(expr == z)
        if data[s.INT_IDX]:
            expr = y[data[s.INT_IDX]]
            z = cp.Variable(shape=(expr.size,), integer=True)
            constraints.append(expr == z)
        return constraints

    def test_lp_2(self):
        sth = STH.lp_2()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_lp_3(self):
        sth = STH.lp_3()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)

    def test_lp_4(self):
        sth = STH.lp_4()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)

    def test_socp_2(self):
        sth = STH.socp_2()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_socp_3(self):
        for axis in [0, 1]:
            sth = STH.socp_3(axis)
            TestSlacks.simulate_chain(sth.prob, [], solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_expcone_1(self):
        sth = STH.expcone_1()
        for affine in TestSlacks.AFF_EXP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_expcone_socp_1(self):
        sth = STH.expcone_socp_1()
        for affine in TestSlacks.AFF_MIXED_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_pcp_1(self):
        sth = STH.pcp_1()
        for affine in TestSlacks.AFF_PCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='SCS', eps=1e-08)
            sth.verify_objective(places=3)
            sth.verify_primal_values(places=3)

    def test_pcp_2(self):
        sth = STH.pcp_2()
        for affine in TestSlacks.AFF_PCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='SCS', eps=1e-08)
            sth.verify_objective(places=3)
            sth.verify_primal_values(places=3)

    def test_mi_lp_1(self):
        sth = STH.mi_lp_1()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS_BB')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    @pytest.mark.skip(reason='Known bug in ECOS BB')
    def test_mi_socp_1(self):
        sth = STH.mi_socp_1()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS_BB')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    @unittest.skipUnless([svr for svr in INSTALLED_MI if svr in MI_SOCP and svr != 'ECOS_BB'], 'No appropriate mixed-integer SOCP solver is installed.')
    def test_mi_socp_2(self):
        sth = STH.mi_socp_2()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine)
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)