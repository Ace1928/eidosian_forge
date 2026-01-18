import numpy as np
import scipy as sp
from cvxpy import settings as s
from cvxpy.constraints.exponential import ExpCone as ExpCone_obj
from cvxpy.constraints.nonpos import NonNeg as NonNeg_obj
from cvxpy.constraints.power import PowCone3D as PowCone_obj
from cvxpy.constraints.psd import PSD as PSD_obj
from cvxpy.constraints.second_order import SOC as SOC_obj
from cvxpy.constraints.zero import Zero as Zero_obj
from cvxpy.reductions.solution import Solution
class Dualize:
    """
    CVXPY represents cone programs as

        (P-Opt) min{ c.T @ x : A @ x + b in K } + d,

    where the corresponding dual is

        (D-Opt) max{ -b @ y : c = A.T @ y, y in K^* } + d.

    For some solvers, it is much easier to specify a problem of the form (D-Opt) than it
    is to specify a problem of the form (P-Opt). The purpose of this reduction is to handle
    mapping between (P-Opt) and (D-Opt) so that a solver interface can pretend the original
    problem was stated in terms (D-Opt).

    Usage
    -----
    Dualize applies to ParamConeProg problems. It accesses (P-Opt) data by calling
    ``c, d, A, b = problem.apply_parameters()``. It assumes the solver interface
    has already executed its ``format_constraints`` function on the ParamConeProg problem.

    A solver interface is responsible for calling both Dualize.apply and Dualize.invert.
    The call to Dualize.apply should be one of the first things that happens, and the
    call to Dualize.invert should be one of the last things that happens.

    The "data" dict returned by Dualize.apply is keyed by s.A, s.B, s.C, and 'K_dir',
    which respectively provide the dual constraint matrix (A.T), the dual constraint
    right-hand-side (c), the dual objective vector (-b), and the dual cones (K^*).
    The solver interface should interpret this data is a new primal problem, just with a
    maximization objective. Given a numerical solution, the solver interface should first
    construct a CVXPY Solution object where :math:`y` is a primal variable, divided into
    several blocks according to the structure of elementary cones appearing in K^*. The only
    dual variable we use is that corresponding to the equality constraint :math:`c = A^T y`.
    No attempt should be made to map unbounded / infeasible status codes for (D-Opt) back
    to unbounded / infeasible status codes for (P-Opt); all such mappings are handled in
    Dualize.invert. Refer to Dualize.invert for detailed documentation.

    Assumptions
    -----------
    The problem has no integer or boolean constraints. This is necessary because strong
    duality does not hold for problems with discrete constraints.

    Dualize.apply assumes "SOLVER.format_constraints()" has already been called. This
    assumption allows flexibility in how a solver interface chooses to vectorize a
    feasible set (e.g. how to order conic constraints, or how to vectorize the PSD cone).

    Additional notes
    ----------------

    Dualize.invert is written in a way which is agnostic to how a solver formats constraints,
    but it also imposes specific requirements on the input. Providing correct input to
    Dualize.invert requires consideration to the effect of ``SOLVER.format_constraints`` and
    the output of ``problem.apply_parameters``.
    """

    @staticmethod
    def apply(problem):
        c, d, A, b = problem.apply_parameters()
        Kp = problem.cone_dims
        Kd = {FREE: Kp.zero, NONNEG: Kp.nonneg, SOC: Kp.soc, PSD: Kp.psd, DUAL_EXP: Kp.exp, DUAL_POW3D: Kp.p3d}
        data = {s.A: A.T, s.B: c, s.C: -b, 'K_dir': Kd, 'dualized': True}
        inv_data = {s.OBJ_OFFSET: d, 'constr_map': problem.constr_map, 'x_id': problem.x.id, 'K_dir': Kd, 'dualized': True}
        return (data, inv_data)

    @staticmethod
    def invert(solution, inv_data):
        """
        ``solution`` is a CVXPY Solution object, formatted where

            (D-Opt) max{ -b @ y : c = A.T @ y, y in K^* } + d

        is the primal problem from the solver's perspective. The purpose of this function
        is to map such a solution back to the format

                (P-Opt) min{ c.T @ x : A @ x + b in K } + d.

        This function handles mapping of primal and dual variables, and solver status codes.
        The variable "x" in (P-Opt) is trivially populated from the dual variables to the
        constraint "c = A.T @ y" in (D-Opt). Status codes also map back in a simple way.

        Details on required formatting of solution.primal_vars
        ------------------------------------------------------

        We assume the dict solution.primal_vars is keyed by string-enums FREE ('fr'), NONNEG ('+'),
        SOC ('s'), PSD ('p'), and DUAL_EXP ('de'). The corresponding values are described below.

        solution.primal_vars[FREE] should be a single vector. It corresponds to the (possibly
        concatenated) components of "y" which are subject to no conic constraints. We map these
        variables back to dual variables for equality constraints in (P-Opt).

        solution.primal_vars[NONNEG] should also be a single vector, this time giving the
        possibly concatenated components of "y" which must be >= 0. We map these variables
        back to dual variables for inequality constraints in (P-Opt).

        solution.primal_vars[SOC] is a list of vectors specifying blocks of "y" which belong
        to the second-order-cone under the CVXPY standard ({ z : z[0] >= || z[1:] || }).
        We map these variables back to dual variables for SOC constraints in (P-Opt).

        solution.primal_vars[PSD] is a list of symmetric positive semidefinite matrices
        which result by lifting the vectorized PSD blocks of "y" back into matrix form.
        We assign these as dual variables to PSD constraints appearing in (P-Opt).

        solution.primal_vars[DUAL_EXP] is a vector of concatenated length-3 slices of y, where
        each constituent length-3 slice belongs to dual exponential cone as implied by the CVXPY
        standard of the primal exponential cone (see cvxpy/constraints/exponential.py:ExpCone).
        We map these back to dual variables for exponential cone constraints in (P-Opt).

        """
        status = solution.status
        prob_attr = solution.attr
        primal_vars, dual_vars = (None, None)
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.opt_val + inv_data[s.OBJ_OFFSET]
            primal_vars = {inv_data['x_id']: solution.dual_vars[s.EQ_DUAL]}
            dual_vars = dict()
            direct_prims = solution.primal_vars
            constr_map = inv_data['constr_map']
            i = 0
            for con in constr_map[Zero_obj]:
                dv = direct_prims[FREE][i:i + con.size]
                dual_vars[con.id] = dv if dv.size > 1 else dv.item()
                i += con.size
            i = 0
            for con in constr_map[NonNeg_obj]:
                dv = direct_prims[NONNEG][i:i + con.size]
                dual_vars[con.id] = dv if dv.size > 1 else dv.item()
                i += con.size
            i = 0
            for con in constr_map[SOC_obj]:
                block_len = con.shape[0]
                dv = np.concatenate(direct_prims[SOC][i:i + block_len])
                dual_vars[con.id] = dv
                i += block_len
            for i, con in enumerate(constr_map[PSD_obj]):
                dv = direct_prims[PSD][i]
                dual_vars[con.id] = dv
            i = 0
            for con in constr_map[ExpCone_obj]:
                dv = direct_prims[DUAL_EXP][i:i + con.size]
                dual_vars[con.id] = dv
                i += con.size
            i = 0
            for con in constr_map[PowCone_obj]:
                dv = direct_prims[DUAL_POW3D][i:i + con.size]
                dual_vars[con.id] = dv
                i += con.size
        elif status == s.INFEASIBLE:
            status = s.UNBOUNDED
            opt_val = -np.inf
        elif status == s.INFEASIBLE_INACCURATE:
            status = s.UNBOUNDED_INACCURATE
            opt_val = -np.inf
        elif status == s.UNBOUNDED:
            status = s.INFEASIBLE
            opt_val = np.inf
        elif status == s.UNBOUNDED_INACCURATE:
            status = s.INFEASIBLE_INACCURATE
            opt_val = np.inf
        else:
            status = s.SOLVER_ERROR
            opt_val = np.nan
        sol = Solution(status, opt_val, primal_vars, dual_vars, prob_attr)
        return sol