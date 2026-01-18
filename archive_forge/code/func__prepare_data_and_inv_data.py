import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import (
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import group_constraints
def _prepare_data_and_inv_data(self, problem):
    data = {}
    inv_data = {self.VAR_ID: problem.x.id}
    constr_map = group_constraints(problem.constraints)
    data[QpSolver.DIMS] = ConeDims(constr_map)
    inv_data[QpSolver.DIMS] = data[QpSolver.DIMS]
    inv_data[QpSolver.IS_MIP] = problem.is_mixed_integer()
    data[s.PARAM_PROB] = problem
    return (problem, data, inv_data)