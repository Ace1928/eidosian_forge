from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
@staticmethod
def _build_dualized_task(task, data):
    """
        This function assumes "data" is formatted according to MOSEK.apply when the problem
        features no integer constraints. This dictionary should contain keys s.C, s.A, s.B,
        'K_dir', 'c_bar_data' and 'A_bar_data'.

        If the problem has no PSD constraints, then we construct a Task representing

           max{ c.T @ x : A @ x == b, x in K_dir }

        If the problem has PSD constraints, then the Task looks like

           max{ c.T @ x + c_bar(X_bars) : A @ x + A_bar(X_bars) == b, x in K_dir, X_bars PSD }

        In the above formulation, c_bar is effectively specified by a list of appropriately
        formatted symmetric matrices (one symmetric matrix for each PSD variable). A_bar
        is specified a collection of symmetric matrix data indexed by (i, j) where the j-th
        PSD variable contributes a certain scalar to the i-th linear equation in the system
        "A @ x + A_bar(X_bars) == b".
        """
    import mosek
    c, A, b, K = (data[s.C], data[s.A], data[s.B], data['K_dir'])
    n, m = A.shape
    task.appendvars(m)
    o = np.zeros(m)
    task.putvarboundlist(np.arange(m, dtype=int), [mosek.boundkey.fr] * m, o, o)
    task.appendcons(n)
    task.putclist(np.arange(c.size, dtype=int), c)
    task.putobjsense(mosek.objsense.maximize)
    rows, cols, vals = sp.sparse.find(A)
    task.putaijlist(rows.tolist(), cols.tolist(), vals.tolist())
    task.putconboundlist(np.arange(n, dtype=int), [mosek.boundkey.fx] * n, b, b)
    idx = K[a2d.FREE]
    num_pos = K[a2d.NONNEG]
    if num_pos > 0:
        o = np.zeros(num_pos)
        task.putvarboundlist(np.arange(idx, idx + num_pos, dtype=int), [mosek.boundkey.lo] * num_pos, o, o)
        idx += num_pos
    num_soc = len(K[a2d.SOC])
    if num_soc > 0:
        cones = [mosek.conetype.quad] * num_soc
        task.appendconesseq(cones, [0] * num_soc, K[a2d.SOC], idx)
        idx += sum(K[a2d.SOC])
    num_dexp = K[a2d.DUAL_EXP]
    if num_dexp > 0:
        cones = [mosek.conetype.dexp] * num_dexp
        task.appendconesseq(cones, [0] * num_dexp, [3] * num_dexp, idx)
        idx += 3 * num_dexp
    num_dpow = len(K[a2d.DUAL_POW3D])
    if num_dpow > 0:
        cones = [mosek.conetype.dpow] * num_dpow
        task.appendconesseq(cones, K[a2d.DUAL_POW3D], [3] * num_dpow, idx)
        idx += 3 * num_dpow
    num_psd = len(K[a2d.PSD])
    if num_psd > 0:
        task.appendbarvars(K[a2d.PSD])
        psd_dims = np.array(K[a2d.PSD])
        for i, j, triples in data['A_bar_data']:
            order = psd_dims[j]
            operator_id = task.appendsparsesymmat(order, triples[0], triples[1], triples[2])
            task.putbaraij(i, j, [operator_id], [1.0])
        for j, triples in data['c_bar_data']:
            order = psd_dims[j]
            operator_id = task.appendsparsesymmat(order, triples[0], triples[1], triples[2])
            task.putbarcj(j, [operator_id], [1.0])
    return task