import numpy as np
from scipy import sparse
import cvxpy as cp
from cvxpy import problems
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.reduction import Reduction
class SOC2PSD(Reduction):
    """Convert all SOC constraints to equivalent PSD constraints.
    """

    def accepts(self, problem):
        return True

    def apply(self, problem):
        soc_constraints = []
        other_constraints = []
        for constraint in problem.constraints:
            if type(constraint) is SOC:
                soc_constraints.append(constraint)
            else:
                other_constraints.append(constraint)
        psd_constraints = []
        soc_constraint_ids = []
        soc_id_from_psd = {}
        for constraint in soc_constraints:
            '\n            The SOC constraint :math:`\\lVert X \\rVert_2 \\leq t` is modeled by `t` and `X`.\n            We extract these `t` and `X` from the SOC constraint object.\n            '
            t, X = constraint.args
            soc_constraint_ids.append(constraint.id)
            '\n            A PSD constraint object will constrain the matrix M specified in its constructor to\n            the PSD cone.\n\n            We will create this matrix M using the `t` and `X` extracted from the SOC constraint.\n\n            Since M being PSD means its Schur complement is also PSD, replacing :math:`M >> 0`\n            with :math:`SchurComplement(M) >> 0` should give us the original SOC constraint.\n            '
            if t.shape == (1,):
                scalar_term = t[0]
                vector_term_len = X.shape[0]
                '\n                We construct the terms A, B and C that comprise the Schur complement of M\n                There are multiple ways to construct A, B and C (and hence M) that are equivalent\n                however, this one makes writing `invert` routine simple.\n                '
                A = scalar_term * sparse.eye(1)
                B = cp.reshape(X, [-1, 1]).T
                C = scalar_term * sparse.eye(vector_term_len)
                '\n                Another technique for reference\n\n                A = scalar_term * sparse.eye(vector_term_len)\n                B = cp.reshape(X,[-1,1])\n                C = scalar_term * sparse.eye(1)\n                '
                '\n                Construct M from A, B and C\n                '
                M = cp.bmat([[A, B], [B.T, C]])
                '\n                Constrain M to the PSD cone.\n                '
                new_psd_constraint = PSD(M)
                soc_id_from_psd[new_psd_constraint.id] = constraint.id
                psd_constraints.append(new_psd_constraint)
            else:
                if constraint.axis == 1:
                    X = X.T
                for subidx in range(t.shape[0]):
                    scalar_term = t[subidx]
                    vector_term_len = X.shape[0]
                    A = scalar_term * sparse.eye(1)
                    B = X[:, subidx:subidx + 1].T
                    C = scalar_term * sparse.eye(vector_term_len)
                    M = cp.bmat([[A, B], [B.T, C]])
                    new_psd_constraint = PSD(M)
                    soc_id_from_psd[new_psd_constraint.id] = constraint.id
                    psd_constraints.append(new_psd_constraint)
        new_problem = problems.problem.Problem(problem.objective, other_constraints + psd_constraints)
        inverse_data = (soc_id_from_psd, soc_constraint_ids)
        return (new_problem, inverse_data)

    def invert(self, solution, inverse_data):
        """
        `solution.dual_vars` contains dual variables corresponding to the constraints.

        The dual variables that we return in `solution` should correspond to the original
        SOC constraints, and not their PSD equivalents. To this end, inversion is required.
        """
        if solution.dual_vars == {}:
            return solution
        soc_id_from_psd, soc_constraint_ids = inverse_data
        psd_constraint_ids = soc_id_from_psd.keys()
        inverted_dual_vars = {}
        for constr_id in soc_constraint_ids:
            inverted_dual_vars[constr_id] = []
        for var_id in solution.dual_vars:
            if var_id in psd_constraint_ids:
                psd_dual_var = solution.dual_vars[var_id]
                soc_dual_var = psd_dual_var[0]
                soc_var_id = soc_id_from_psd[var_id]
                inverted_dual_vars[soc_var_id].append(soc_dual_var)
            else:
                inverted_dual_vars[var_id] = solution.dual_vars[var_id]
        for var_id in inverted_dual_vars:
            if var_id in soc_constraint_ids:
                inverted_dual_vars[var_id] = 2 * np.hstack(inverted_dual_vars[var_id])
        solution.dual_vars = inverted_dual_vars
        return solution