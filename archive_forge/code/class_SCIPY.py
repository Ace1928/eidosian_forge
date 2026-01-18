import warnings
import numpy as np
import scipy  # For version checks
import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
class SCIPY(ConicSolver):
    """An interface for the SciPy linprog function.
    Note: This requires a version of SciPy which is >= 1.6.1
    """
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    if Version(scipy.__version__) < Version('1.9.0'):
        MIP_CAPABLE = False
    else:
        MIP_CAPABLE = True
        MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS
    STATUS_MAP = {0: s.OPTIMAL, 1: s.OPTIMAL_INACCURATE, 2: s.INFEASIBLE, 3: s.UNBOUNDED, 4: s.SOLVER_ERROR}

    def import_solver(self) -> None:
        """Imports the solver.
        """
        from scipy import optimize as opt

    def name(self):
        """The name of the solver.
        """
        return s.SCIPY

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}
        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]
        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg]
        len_eq = problem.cone_dims.zero
        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A[:len_eq]
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = b[:len_eq].flatten()
        if data[s.B].shape[0] == 0:
            data[s.B] = None
        data[s.G] = -A[len_eq:]
        if 0 in data[s.G].shape:
            data[s.G] = None
        data[s.H] = b[len_eq:].flatten()
        if 0 in data[s.H].shape:
            data[s.H] = None
        return (data, inv_data)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        from scipy import optimize as opt
        if Version(scipy.__version__) < Version('1.6.1'):
            meth = 'interior-point'
        else:
            meth = 'highs'
        problem_is_a_mip = data[s.BOOL_IDX] or data[s.INT_IDX]
        if problem_is_a_mip:
            integrality = [0] * data[s.C].shape[0]
            for index in data[s.BOOL_IDX] + data[s.INT_IDX]:
                integrality[index] = 1
            bounds = [(None, None)] * data[s.C].shape[0]
            for index in data[s.BOOL_IDX]:
                bounds[index] = (0, 1)
        else:
            integrality = None
            bounds = (None, None)
        if solver_opts:
            if 'scipy_options' not in solver_opts:
                raise ValueError("All parameters for the SCIPY solver should be encased within a dictionary called scipy_options e.g. \nprob.solve(solver='SCIPY', verbose=True, scipy_options={'method':'highs-ds', 'maxiter':10000})")
            if Version(scipy.__version__) < Version('1.9.0'):
                if 'method' not in solver_opts['scipy_options']:
                    self._log_scipy_method_warning(meth)
            if 'method' in solver_opts['scipy_options']:
                meth = solver_opts['scipy_options'].pop('method')
                ver = Version(scipy.__version__) < Version('1.6.1')
                if (meth in ['highs-ds', 'highs-ipm', 'highs']) & ver:
                    raise ValueError('The HiGHS solvers require a SciPy version >= 1.6.1')
            if 'bounds' in solver_opts['scipy_options']:
                raise ValueError('Please do not specify bounds through scipy_options. Please specify bounds through CVXPY.')
            if 'integrality' in solver_opts['scipy_options']:
                raise ValueError('Please do not specify variable integrality through scipy_options. Please specify variable types through CVXPY.')
            method_supports_mip = meth == 'highs'
            if problem_is_a_mip and (not method_supports_mip):
                raise ValueError("Only the 'highs' SciPy method can solve MIP models.")
        else:
            solver_opts['scipy_options'] = {}
            if Version(scipy.__version__) < Version('1.9.0'):
                self._log_scipy_method_warning(meth)
        if problem_is_a_mip:
            constraints = []
            G = data[s.G]
            if G is not None:
                ineq = scipy.optimize.LinearConstraint(G, ub=data[s.H])
                constraints.append(ineq)
            A = data[s.A]
            if A is not None:
                eq = scipy.optimize.LinearConstraint(A, data[s.B], data[s.B])
                constraints.append(eq)
            lb = [t[0] if t[0] is not None else -np.inf for t in bounds]
            ub = [t[1] if t[1] is not None else np.inf for t in bounds]
            bounds = scipy.optimize.Bounds(lb, ub)
            solution = opt.milp(data[s.C], constraints=constraints, options=solver_opts['scipy_options'], integrality=integrality, bounds=bounds)
        else:
            solution = opt.linprog(data[s.C], A_ub=data[s.G], b_ub=data[s.H], A_eq=data[s.A], b_eq=data[s.B], method=meth, options=solver_opts['scipy_options'], bounds=bounds)
        solver_opts['scipy_options']['method'] = meth
        if verbose is True:
            print('Solver terminated with message: ' + solution.message)
        return solution

    def _log_scipy_method_warning(self, meth):
        warnings.warn("It is best to specify the 'method' parameter within scipy_options. The main advantage of this solver is its ability to use the HiGHS LP solvers via scipy.optimize.linprog(), which requires a SciPy version >= 1.6.1.\n\nThe default method '{}' will be used in this case.\n".format(meth))

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = self.STATUS_MAP[solution['status']]
        if status == s.OPTIMAL_INACCURATE and solution.x is None:
            status = s.SOLVER_ERROR
        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['fun']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['x']}
            if 'ineqlin' in solution and (not inverse_data['is_mip']):
                eq_dual = utilities.get_dual_values(-solution['eqlin']['marginals'], utilities.extract_dual_value, inverse_data[self.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(-solution['ineqlin']['marginals'], utilities.extract_dual_value, inverse_data[self.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            attr = {}
            if 'mip_gap' in solution:
                attr[s.EXTRA_STATS] = {'mip_gap': solution['mip_gap']}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status)