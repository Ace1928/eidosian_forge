from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.solver import Solver
class ConicSolver(Solver):
    """Conic solver class with reduction semantics
    """
    DIMS = 'dims'
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]
    REQUIRES_CONSTR = False
    EXP_CONE_ORDER = None

    def supports_quad_obj(self) -> bool:
        """By default does not support a quadratic objective.
        """
        return False

    def accepts(self, problem):
        return isinstance(problem, ParamConeProg) and (self.MIP_CAPABLE or not problem.is_mixed_integer()) and (not convex_attributes([problem.x])) and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR) and all((type(c) in self.SUPPORTED_CONSTRAINTS for c in problem.constraints))

    @staticmethod
    def get_spacing_matrix(shape: Tuple[int, ...], spacing, streak, num_blocks, offset):
        """Returns a sparse matrix that spaces out an expression.

        Parameters
        ----------
        shape : tuple
            (rows in matrix, columns in matrix)
        spacing : int
            The number of rows between the start of each non-zero block.
        streak: int
            The number of elements in each block.
        num_blocks : int
            The number of non-zero blocks.
        offset : int
            The number of zero rows at the beginning of the matrix.

        Returns
        -------
        SciPy CSC matrix
            A sparse matrix
        """
        num_values = num_blocks * streak
        val_arr = np.ones(num_values, dtype=np.float64)
        streak_plus_spacing = streak + spacing
        row_arr = np.arange(0, num_blocks * streak_plus_spacing).reshape(num_blocks, streak_plus_spacing)[:, :streak].flatten() + offset
        col_arr = np.arange(num_values)
        return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)

    @staticmethod
    def psd_format_mat(constr):
        """Return a matrix to multiply by PSD constraint coefficients.
        """
        return sp.eye(constr.size, format='csc')

    def format_constraints(self, problem, exp_cone_order):
        """
        Returns a ParamConeProg whose problem data tensors will yield the
        coefficient "A" and offset "b" for the constraint in the following
        formats:
            Linear equations: (A, b) such that A * x + b == 0,
            Linear inequalities: (A, b) such that A * x + b >= 0,
            Second order cone: (A, b) such that A * x + b in SOC,
            Exponential cone: (A, b) such that A * x + b in EXP,
            Semidefinite cone: (A, b) such that A * x + b in PSD,

        The CVXPY standard for the exponential cone is:
            K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
        Whenever a solver uses this convention, EXP_CONE_ORDER should be
        [0, 1, 2].

        The CVXPY standard for the second order cone is:
            SOC(n) = { x : x[0] >= norm(x[1:n], 2)  }.
        All currently supported solvers use this convention.

        Args:
          problem : ParamConeProg
            The problem that is the provenance of the constraint.
          exp_cone_order: list
            A list indicating how the exponential cone arguments are ordered.

        Returns:
          ParamConeProg with structured A.
        """
        restruct_mat = []
        for constr in problem.constraints:
            total_height = sum([arg.size for arg in constr.args])
            if type(constr) == Zero:
                restruct_mat.append(NegativeIdentityOperator(constr.size))
            elif type(constr) == NonNeg:
                restruct_mat.append(IdentityOperator(constr.size))
            elif type(constr) == SOC:
                assert constr.axis == 0, 'SOC must be lowered to axis == 0'
                t_spacer = ConicSolver.get_spacing_matrix(shape=(total_height, constr.args[0].size), spacing=constr.args[1].shape[0], streak=1, num_blocks=constr.args[0].size, offset=0)
                X_spacer = ConicSolver.get_spacing_matrix(shape=(total_height, constr.args[1].size), spacing=1, streak=constr.args[1].shape[0], num_blocks=constr.args[0].size, offset=1)
                restruct_mat.append(sp.hstack([t_spacer, X_spacer]))
            elif type(constr) == ExpCone:
                arg_mats = []
                for i, arg in enumerate(constr.args):
                    space_mat = ConicSolver.get_spacing_matrix(shape=(total_height, arg.size), spacing=len(exp_cone_order) - 1, streak=1, num_blocks=arg.size, offset=exp_cone_order[i])
                    arg_mats.append(space_mat)
                restruct_mat.append(sp.hstack(arg_mats))
            elif type(constr) == PowCone3D:
                arg_mats = []
                for i, arg in enumerate(constr.args):
                    space_mat = ConicSolver.get_spacing_matrix(shape=(total_height, arg.size), spacing=2, streak=1, num_blocks=arg.size, offset=i)
                    arg_mats.append(space_mat)
                restruct_mat.append(sp.hstack(arg_mats))
            elif type(constr) == PSD:
                restruct_mat.append(self.psd_format_mat(constr))
            else:
                raise ValueError('Unsupported constraint type.')
        if restruct_mat:
            restruct_mat = as_block_diag_linear_operator(restruct_mat)
            unspecified, remainder = divmod(problem.A.shape[0] * problem.A.shape[1], restruct_mat.shape[1])
            reshaped_A = problem.A.reshape(restruct_mat.shape[1], unspecified, order='F').tocsr()
            restructured_A = restruct_mat(reshaped_A).tocoo()
            restructured_A.row = restructured_A.row.astype(np.int64)
            restructured_A.col = restructured_A.col.astype(np.int64)
            restructured_A = restructured_A.reshape(np.int64(restruct_mat.shape[0]) * (np.int64(problem.x.size) + 1), problem.A.shape[1], order='F')
        else:
            restructured_A = problem.A
        new_param_cone_prog = ParamConeProg(problem.c, problem.x, restructured_A, problem.variables, problem.var_id_to_col, problem.constraints, problem.parameters, problem.param_id_to_col, P=problem.P, formatted=True)
        return new_param_cone_prog

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            eq_dual = utilities.get_dual_values(solution['eq_dual'], utilities.extract_dual_value, inverse_data[Solver.EQ_CONSTR])
            leq_dual = utilities.get_dual_values(solution['ineq_dual'], utilities.extract_dual_value, inverse_data[Solver.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def _prepare_data_and_inv_data(self, problem):
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims
        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[SOC] + constr_map[PSD] + constr_map[ExpCone] + constr_map[PowCone3D]
        return (problem, data, inv_data)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)
        if problem.P is None:
            c, d, A, b = problem.apply_parameters()
        else:
            P, c, d, A, b = problem.apply_parameters(quad_obj=True)
            data[s.P] = P
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A
        data[s.B] = b
        return (data, inv_data)