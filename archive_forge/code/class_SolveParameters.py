import dataclasses
import datetime
import enum
from typing import Dict, Optional
from ortools.pdlp import solvers_pb2 as pdlp_solvers_pb2
from ortools.glop import parameters_pb2 as glop_parameters_pb2
from ortools.gscip import gscip_pb2
from ortools.math_opt import parameters_pb2 as math_opt_parameters_pb2
from ortools.math_opt.solvers import glpk_pb2
from ortools.math_opt.solvers import gurobi_pb2
from ortools.math_opt.solvers import highs_pb2
from ortools.math_opt.solvers import osqp_pb2
from ortools.sat import sat_parameters_pb2
@dataclasses.dataclass
class SolveParameters:
    """Parameters to control a single solve.

  If a value is set in both common and solver specific field (e.g. gscip), the
  solver specific setting is used.

  Solver specific parameters for solvers other than the one in use are ignored.

  Parameters that depends on the model (e.g. branching priority is set for each
  variable) are passed in ModelSolveParameters.

  See solve() and IncrementalSolver.solve() in solve.py for more details.

  Attributes:
    time_limit: The maximum time a solver should spend on the problem, or if
      None, then the time limit is infinite. This value is not a hard limit,
      solve time may slightly exceed this value. This parameter is always passed
      to the underlying solver, the solver default is not used.
    iteration_limit: Limit on the iterations of the underlying algorithm (e.g.
      simplex pivots). The specific behavior is dependent on the solver and
      algorithm used, but often can give a deterministic solve limit (further
      configuration may be needed, e.g. one thread). Typically supported by LP,
      QP, and MIP solvers, but for MIP solvers see also node_limit.
    node_limit: Limit on the number of subproblems solved in enumerative search
      (e.g. branch and bound). For many solvers this can be used to
      deterministically limit computation (further configuration may be needed,
      e.g. one thread). Typically for MIP solvers, see also iteration_limit.
    cutoff_limit: The solver stops early if it can prove there are no primal
      solutions at least as good as cutoff. On an early stop, the solver returns
      TerminationReason.NO_SOLUTION_FOUND and with Limit.CUTOFF and is not
      required to give any extra solution information. Has no effect on the
      return value if there is no early stop. It is recommended that you use a
      tolerance if you want solutions with objective exactly equal to cutoff to
      be returned. See the user guide for more details and a comparison with
      best_bound_limit.
    objective_limit: The solver stops early as soon as it finds a solution at
      least this good, with TerminationReason.FEASIBLE and Limit.OBJECTIVE.
    best_bound_limit: The solver stops early as soon as it proves the best bound
      is at least this good, with TerminationReason of FEASIBLE or
      NO_SOLUTION_FOUND and Limit.OBJECTIVE. See the user guide for more details
      and a comparison with cutoff_limit.
    solution_limit: The solver stops early after finding this many feasible
      solutions, with TerminationReason.FEASIBLE and Limit.SOLUTION. Must be
      greater than zero if set. It is often used get the solver to stop on the
      first feasible solution found. Note that there is no guarantee on the
      objective value for any of the returned solutions. Solvers will typically
      not return more solutions than the solution limit, but this is not
      enforced by MathOpt, see also b/214041169. Currently supported for Gurobi
      and SCIP, and for CP-SAT only with value 1.
    enable_output: If the solver should print out its log messages.
    threads: An integer >= 1, how many threads to use when solving.
    random_seed: Seed for the pseudo-random number generator in the underlying
      solver. Note that valid values depend on the actual solver:
        * Gurobi: [0:GRB_MAXINT] (which as of Gurobi 9.0 is 2x10^9).
        * GSCIP: [0:2147483647] (which is MAX_INT or kint32max or 2^31-1).
        * GLOP: [0:2147483647] (same as above).
      In all cases, the solver will receive a value equal to:
      MAX(0, MIN(MAX_VALID_VALUE_FOR_SOLVER, random_seed)).
    absolute_gap_tolerance: An absolute optimality tolerance (primarily) for MIP
      solvers. The absolute GAP is the absolute value of the difference between:
        * the objective value of the best feasible solution found,
        * the dual bound produced by the search.
      The solver can stop once the absolute GAP is at most
      absolute_gap_tolerance (when set), and return TerminationReason.OPTIMAL.
      Must be >= 0 if set. See also relative_gap_tolerance.
    relative_gap_tolerance: A relative optimality tolerance (primarily) for MIP
      solvers. The relative GAP is a normalized version of the absolute GAP
      (defined on absolute_gap_tolerance), where the normalization is
      solver-dependent, e.g. the absolute GAP divided by the objective value of
      the best feasible solution found. The solver can stop once the relative
      GAP is at most relative_gap_tolerance (when set), and return
      TerminationReason.OPTIMAL. Must be >= 0 if set. See also
      absolute_gap_tolerance.
    solution_pool_size: Maintain up to `solution_pool_size` solutions while
      searching. The solution pool generally has two functions:
        * For solvers that can return more than one solution, this limits how
          many solutions will be returned.
        * Some solvers may run heuristics using solutions from the solution
          pool, so changing this value may affect the algorithm's path.
      To force the solver to fill the solution pool, e.g. with the n best
      solutions, requires further, solver specific configuration.
    lp_algorithm: The algorithm for solving a linear program. If UNSPECIFIED,
      use the solver default algorithm. For problems that are not linear
      programs but where linear programming is a subroutine, solvers may use
      this value. E.g. MIP solvers will typically use this for the root LP solve
      only (and use dual simplex otherwise).
    presolve: Effort on simplifying the problem before starting the main
      algorithm (e.g. simplex).
    cuts: Effort on getting a stronger LP relaxation (MIP only). Note that in
      some solvers, disabling cuts may prevent callbacks from having a chance to
      add cuts at MIP_NODE.
    heuristics: Effort in finding feasible solutions beyond those encountered in
      the complete search procedure.
    scaling: Effort in rescaling the problem to improve numerical stability.
    gscip: GSCIP specific solve parameters.
    gurobi: Gurobi specific solve parameters.
    glop: Glop specific solve parameters.
    cp_sat: CP-SAT specific solve parameters.
    pdlp: PDLP specific solve parameters.
    osqp: OSQP specific solve parameters. Users should prefer the generic
      MathOpt parameters over OSQP-level parameters, when available: - Prefer
      SolveParameters.enable_output to OsqpSettingsProto.verbose. - Prefer
      SolveParameters.time_limit to OsqpSettingsProto.time_limit. - Prefer
      SolveParameters.iteration_limit to OsqpSettingsProto.iteration_limit. - If
      a less granular configuration is acceptable, prefer
      SolveParameters.scaling to OsqpSettingsProto.
    glpk: GLPK specific solve parameters.
    highs: HiGHS specific solve parameters.
  """
    time_limit: Optional[datetime.timedelta] = None
    iteration_limit: Optional[int] = None
    node_limit: Optional[int] = None
    cutoff_limit: Optional[float] = None
    objective_limit: Optional[float] = None
    best_bound_limit: Optional[float] = None
    solution_limit: Optional[int] = None
    enable_output: bool = False
    threads: Optional[int] = None
    random_seed: Optional[int] = None
    absolute_gap_tolerance: Optional[float] = None
    relative_gap_tolerance: Optional[float] = None
    solution_pool_size: Optional[int] = None
    lp_algorithm: Optional[LPAlgorithm] = None
    presolve: Optional[Emphasis] = None
    cuts: Optional[Emphasis] = None
    heuristics: Optional[Emphasis] = None
    scaling: Optional[Emphasis] = None
    gscip: gscip_pb2.GScipParameters = dataclasses.field(default_factory=gscip_pb2.GScipParameters)
    gurobi: GurobiParameters = dataclasses.field(default_factory=GurobiParameters)
    glop: glop_parameters_pb2.GlopParameters = dataclasses.field(default_factory=glop_parameters_pb2.GlopParameters)
    cp_sat: sat_parameters_pb2.SatParameters = dataclasses.field(default_factory=sat_parameters_pb2.SatParameters)
    pdlp: pdlp_solvers_pb2.PrimalDualHybridGradientParams = dataclasses.field(default_factory=pdlp_solvers_pb2.PrimalDualHybridGradientParams)
    osqp: osqp_pb2.OsqpSettingsProto = dataclasses.field(default_factory=osqp_pb2.OsqpSettingsProto)
    glpk: GlpkParameters = dataclasses.field(default_factory=GlpkParameters)
    highs: highs_pb2.HighsOptionsProto = dataclasses.field(default_factory=highs_pb2.HighsOptionsProto)

    def to_proto(self) -> math_opt_parameters_pb2.SolveParametersProto:
        """Returns a protocol buffer equivalent to this."""
        result = math_opt_parameters_pb2.SolveParametersProto(enable_output=self.enable_output, lp_algorithm=lp_algorithm_to_proto(self.lp_algorithm), presolve=emphasis_to_proto(self.presolve), cuts=emphasis_to_proto(self.cuts), heuristics=emphasis_to_proto(self.heuristics), scaling=emphasis_to_proto(self.scaling), gscip=self.gscip, gurobi=self.gurobi.to_proto(), glop=self.glop, cp_sat=self.cp_sat, pdlp=self.pdlp, osqp=self.osqp, glpk=self.glpk.to_proto(), highs=self.highs)
        if self.time_limit is not None:
            result.time_limit.FromTimedelta(self.time_limit)
        if self.iteration_limit is not None:
            result.iteration_limit = self.iteration_limit
        if self.node_limit is not None:
            result.node_limit = self.node_limit
        if self.cutoff_limit is not None:
            result.cutoff_limit = self.cutoff_limit
        if self.objective_limit is not None:
            result.objective_limit = self.objective_limit
        if self.best_bound_limit is not None:
            result.best_bound_limit = self.best_bound_limit
        if self.solution_limit is not None:
            result.solution_limit = self.solution_limit
        if self.threads is not None:
            result.threads = self.threads
        if self.random_seed is not None:
            result.random_seed = self.random_seed
        if self.absolute_gap_tolerance is not None:
            result.absolute_gap_tolerance = self.absolute_gap_tolerance
        if self.relative_gap_tolerance is not None:
            result.relative_gap_tolerance = self.relative_gap_tolerance
        if self.solution_pool_size is not None:
            result.solution_pool_size = self.solution_pool_size
        return result