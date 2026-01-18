import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.utilities.versioning import Version
class SCS(ConicSolver):
    """An interface for the SCS solver.
    """
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PSD, PowCone3D]
    REQUIRES_CONSTR = True
    STATUS_MAP = {1: s.OPTIMAL, 2: s.OPTIMAL_INACCURATE, -1: s.UNBOUNDED, -6: s.UNBOUNDED_INACCURATE, -2: s.INFEASIBLE, -7: s.INFEASIBLE_INACCURATE, -4: s.SOLVER_ERROR, -3: s.SOLVER_ERROR, -5: s.SOLVER_ERROR}
    EXP_CONE_ORDER = [0, 1, 2]
    ACCELERATION_RETRY_MESSAGE = '\n    CVXPY has just called the numerical solver SCS (version %s),\n    which could not accurately solve the problem with the provided solver\n    options. No value was specified for the SCS option called\n    "acceleration_lookback". That option often has a major impact on\n    whether this version of SCS converges to an accurate solution.\n\n    We will try to solve the problem again by setting acceleration_lookback = 0.\n    To avoid this error in the future we recommend installing SCS version 3.0\n    or higher.\n\n    More information on SCS options can be found at the following URL:\n    https://www.cvxgrp.org/scs/api/settings.html\n    '

    def name(self):
        """The name of the solver.
        """
        return s.SCS

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import scs

    def supports_quad_obj(self) -> bool:
        """SCS >= 3.0.0 supports a quadratic objective.
        """
        import scs
        return Version(scs.__version__) >= Version('3.0.0')

    @staticmethod
    def psd_format_mat(constr):
        """Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as SCS expects constraints to be
        imposed on solely the lower triangular part of the variable matrix.
        Moreover, it requires the off-diagonal coefficients to be scaled by
        sqrt(2), and applies to the symmetric part of the constrained expression.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2
        row_arr = np.arange(0, entries)
        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices, (rows, cols), order='F'))
        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]
        shape = (entries, rows * cols)
        scaled_lower_tri = sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)
        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_matrix((val_symm, (row_symm, col_symm)))
        return scaled_lower_tri @ symm_matrix

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        return super(SCS, self).apply(problem)

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.

        Special cases PSD constraints, as per the SCS specification.
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            lower_tri_dim = dim * (dim + 1) // 2
            new_offset = offset + lower_tri_dim
            lower_tri = result_vec[offset:new_offset]
            full = tri_to_full(lower_tri, dim)
            return (full, new_offset)
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        import scs
        attr = {}
        if Version(scs.__version__) < Version('3.0.0'):
            status = self.STATUS_MAP[solution['info']['statusVal']]
            attr[s.SOLVE_TIME] = solution['info']['solveTime'] / 1000
            attr[s.SETUP_TIME] = solution['info']['setupTime'] / 1000
        else:
            status = self.STATUS_MAP[solution['info']['status_val']]
            attr[s.SOLVE_TIME] = solution['info']['solve_time'] / 1000
            attr[s.SETUP_TIME] = solution['info']['setup_time'] / 1000
        attr[s.NUM_ITERS] = solution['info']['iter']
        attr[s.EXTRA_STATS] = solution
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pobj']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[SCS.VAR_ID]: solution['x']}
            eq_dual_vars = utilities.get_dual_values(solution['y'][:inverse_data[ConicSolver.DIMS].zero], self.extract_dual_value, inverse_data[SCS.EQ_CONSTR])
            ineq_dual_vars = utilities.get_dual_values(solution['y'][inverse_data[ConicSolver.DIMS].zero:], self.extract_dual_value, inverse_data[SCS.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def parse_solver_options(solver_opts):
        import scs
        if Version(scs.__version__) < Version('3.0.0'):
            if 'eps_abs' in solver_opts or 'eps_rel' in solver_opts:
                solver_opts['eps'] = min(solver_opts.get('eps_abs', 1), solver_opts.get('eps_rel', 1))
            else:
                solver_opts['eps'] = solver_opts.get('eps', 0.0001)
        elif 'eps' in solver_opts:
            solver_opts['eps_abs'] = solver_opts['eps']
            solver_opts['eps_rel'] = solver_opts['eps']
            del solver_opts['eps']
        else:
            solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-05)
            solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-05)
        if 'use_quad_obj' in solver_opts:
            del solver_opts['use_quad_obj']
        return solver_opts

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start SCS.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            SCS-specific solver options.

        Returns
        -------
        The result returned by a call to scs.solve().
        """
        import scs
        scs_version = Version(scs.__version__)
        args = {'A': data[s.A], 'b': data[s.B], 'c': data[s.C]}
        if s.P in data:
            args['P'] = data[s.P]
        if warm_start and solver_cache is not None and (self.name() in solver_cache):
            args['x'] = solver_cache[self.name()]['x']
            args['y'] = solver_cache[self.name()]['y']
            args['s'] = solver_cache[self.name()]['s']
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])

        def solve(_solver_opts):
            if scs_version.major < 3:
                _results = scs.solve(args, cones, verbose=verbose, **_solver_opts)
                _status = self.STATUS_MAP[_results['info']['statusVal']]
            else:
                _results = scs.solve(args, cones, verbose=verbose, **_solver_opts)
                _status = self.STATUS_MAP[_results['info']['status_val']]
            return (_results, _status)
        solver_opts = SCS.parse_solver_options(solver_opts)
        results, status = solve(solver_opts)
        if status in s.INACCURATE and scs_version.major == 2 and ('acceleration_lookback' not in solver_opts):
            import warnings
            warnings.warn(SCS.ACCELERATION_RETRY_MESSAGE % str(scs_version))
            retry_opts = solver_opts.copy()
            retry_opts['acceleration_lookback'] = 0
            results, status = solve(retry_opts)
        if solver_cache is not None and status == s.OPTIMAL:
            solver_cache[self.name()] = results
        return results