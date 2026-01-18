import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@dataclasses.dataclass
class SolveResult:
    """The result of solving an optimization problem defined by a Model.

    We attempt to return as much solution information (primal_solutions,
    primal_rays, dual_solutions, dual_rays) as each underlying solver will provide
    given its return status. Differences in the underlying solvers result in a
    weak contract on what fields will be populated for a given termination
    reason. This is discussed in detail in termination_reasons.md, and the most
    important points are summarized below:
      * When the termination reason is optimal, there will be at least one primal
        solution provided that will be feasible up to the underlying solver's
        tolerances.
      * Dual solutions are only given for convex optimization problems (e.g.
        linear programs, not integer programs).
      * A basis is only given for linear programs when solved by the simplex
        method (e.g., not with PDLP).
      * Solvers have widely varying support for returning primal and dual rays.
        E.g. a termination_reason of unbounded does not ensure that a feasible
        solution or a primal ray is returned, check termination_reasons.md for
        solver specific guarantees if this is needed. Further, many solvers will
        provide the ray but not the feasible solution when returning an unbounded
        status.
      * When the termination reason is that a limit was reached or that the result
        is imprecise, a solution may or may not be present. Further, for some
        solvers (generally, convex optimization solvers, not MIP solvers), the
        primal or dual solution may not be feasible.

    Solver specific output is also returned for some solvers (and only information
    for the solver used will be populated).

    Attributes:
      termination: The reason the solver stopped.
      solve_stats: Statistics on the solve process, e.g. running time, iterations.
      solutions: Lexicographically by primal feasibility status, dual feasibility
        status, (basic dual feasibility for simplex solvers), primal objective
        value and dual objective value.
      primal_rays: Directions of unbounded primal improvement, or equivalently,
        dual infeasibility certificates. Typically provided for terminal reasons
        UNBOUNDED and DUAL_INFEASIBLE.
      dual_rays: Directions of unbounded dual improvement, or equivalently, primal
        infeasibility certificates. Typically provided for termination reason
        INFEASIBLE.
      gscip_specific_output: statistics returned by the gSCIP solver, if used.
      osqp_specific_output: statistics returned by the OSQP solver, if used.
      pdlp_specific_output: statistics returned by the PDLP solver, if used.
    """
    termination: Termination = dataclasses.field(default_factory=Termination)
    solve_stats: SolveStats = dataclasses.field(default_factory=SolveStats)
    solutions: List[solution.Solution] = dataclasses.field(default_factory=list)
    primal_rays: List[solution.PrimalRay] = dataclasses.field(default_factory=list)
    dual_rays: List[solution.DualRay] = dataclasses.field(default_factory=list)
    gscip_specific_output: Optional[gscip_pb2.GScipOutput] = None
    osqp_specific_output: Optional[osqp_pb2.OsqpOutput] = None
    pdlp_specific_output: Optional[result_pb2.SolveResultProto.PdlpOutput] = None

    def solve_time(self) -> datetime.timedelta:
        """Shortcut for SolveResult.solve_stats.solve_time."""
        return self.solve_stats.solve_time

    def primal_bound(self) -> float:
        """Returns a primal bound on the optimal objective value as described in ObjectiveBounds.

        Will return a valid (possibly infinite) bound even if no primal feasible
        solutions are available.
        """
        return self.termination.objective_bounds.primal_bound

    def dual_bound(self) -> float:
        """Returns a dual bound on the optimal objective value as described in ObjectiveBounds.

        Will return a valid (possibly infinite) bound even if no dual feasible
        solutions are available.
        """
        return self.termination.objective_bounds.dual_bound

    def has_primal_feasible_solution(self) -> bool:
        """Indicates if at least one primal feasible solution is available.

        When termination.reason is TerminationReason.OPTIMAL or
        TerminationReason.FEASIBLE, this is guaranteed to be true and need not be
        checked.

        Returns:
          True if there is at least one primal feasible solution is available,
          False, otherwise.
        """
        if not self.solutions:
            return False
        return self.solutions[0].primal_solution is not None and self.solutions[0].primal_solution.feasibility_status == solution.SolutionStatus.FEASIBLE

    def objective_value(self) -> float:
        """Returns the objective value of the best primal feasible solution.

        An error will be raised if there are no primal feasible solutions.
        primal_bound() above is guaranteed to be at least as good (larger or equal
        for max problems and smaller or equal for min problems) as objective_value()
        and will never raise an error, so it may be preferable in some cases. Note
        that primal_bound() could be better than objective_value() even for optimal
        terminations, but on such optimal termination, both should satisfy the
        optimality tolerances.

         Returns:
           The objective value of the best primal feasible solution.

         Raises:
           ValueError: There are no primal feasible solutions.
        """
        if not self.has_primal_feasible_solution():
            raise ValueError('No primal feasible solution available.')
        assert self.solutions[0].primal_solution is not None
        return self.solutions[0].primal_solution.objective_value

    def best_objective_bound(self) -> float:
        """Returns a bound on the best possible objective value.

        best_objective_bound() is always equal to dual_bound(), so they can be
        used interchangeably.
        """
        return self.termination.objective_bounds.dual_bound

    @overload
    def variable_values(self, variables: None=...) -> Dict[model.Variable, float]:
        ...

    @overload
    def variable_values(self, variables: model.Variable) -> float:
        ...

    @overload
    def variable_values(self, variables: Iterable[model.Variable]) -> List[float]:
        ...

    def variable_values(self, variables=None):
        """The variable values from the best primal feasible solution.

        An error will be raised if there are no primal feasible solutions.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            variable values to return. If not provided, variable_values returns a
            dictionary with all the variable values for all variables.

        Returns:
          The variable values from the best primal feasible solution.

        Raises:
          ValueError: There are no primal feasible solutions.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
        if not self.has_primal_feasible_solution():
            raise ValueError('No primal feasible solution available.')
        assert self.solutions[0].primal_solution is not None
        if variables is None:
            return self.solutions[0].primal_solution.variable_values
        if isinstance(variables, model.Variable):
            return self.solutions[0].primal_solution.variable_values[variables]
        if isinstance(variables, Iterable):
            return [self.solutions[0].primal_solution.variable_values[v] for v in variables]
        raise TypeError(f'unsupported type in argument for variable_values: {type(variables).__name__!r}')

    def bounded(self) -> bool:
        """Returns true only if the problem has been shown to be feasible and bounded."""
        return self.termination.problem_status.primal_status == FeasibilityStatus.FEASIBLE and self.termination.problem_status.dual_status == FeasibilityStatus.FEASIBLE

    def has_ray(self) -> bool:
        """Indicates if at least one primal ray is available.

        This is NOT guaranteed to be true when termination.reason is
        TerminationReason.kUnbounded or TerminationReason.kInfeasibleOrUnbounded.

        Returns:
          True if at least one primal ray is available.
        """
        return bool(self.primal_rays)

    @overload
    def ray_variable_values(self, variables: None=...) -> Dict[model.Variable, float]:
        ...

    @overload
    def ray_variable_values(self, variables: model.Variable) -> float:
        ...

    @overload
    def ray_variable_values(self, variables: Iterable[model.Variable]) -> List[float]:
        ...

    def ray_variable_values(self, variables=None):
        """The variable values from the first primal ray.

        An error will be raised if there are no primal rays.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            variable values to return. If not provided, variable_values() returns a
            dictionary with the variable values for all variables.

        Returns:
          The variable values from the first primal ray.

        Raises:
          ValueError: There are no primal rays.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
        if not self.has_ray():
            raise ValueError('No primal ray available.')
        if variables is None:
            return self.primal_rays[0].variable_values
        if isinstance(variables, model.Variable):
            return self.primal_rays[0].variable_values[variables]
        if isinstance(variables, Iterable):
            return [self.primal_rays[0].variable_values[v] for v in variables]
        raise TypeError(f'unsupported type in argument for ray_variable_values: {type(variables).__name__!r}')

    def has_dual_feasible_solution(self) -> bool:
        """Indicates if the best solution has an associated dual feasible solution.

        This is NOT guaranteed to be true when termination.reason is
        TerminationReason.Optimal. It also may be true even when the best solution
        does not have an associated primal feasible solution.

        Returns:
          True if the best solution has an associated dual feasible solution.
        """
        if not self.solutions:
            return False
        return self.solutions[0].dual_solution is not None and self.solutions[0].dual_solution.feasibility_status == solution.SolutionStatus.FEASIBLE

    @overload
    def dual_values(self, linear_constraints: None=...) -> Dict[model.LinearConstraint, float]:
        ...

    @overload
    def dual_values(self, linear_constraints: model.LinearConstraint) -> float:
        ...

    @overload
    def dual_values(self, linear_constraints: Iterable[model.LinearConstraint]) -> List[float]:
        ...

    def dual_values(self, linear_constraints=None):
        """The dual values associated to the best solution.

        If there is at least one primal feasible solution, this corresponds to the
        dual values associated to the best primal feasible solution. An error will
        be raised if the best solution does not have an associated dual feasible
        solution.

        Args:
          linear_constraints: an optional LinearConstraint or iterator of
            LinearConstraint indicating what dual values to return. If not provided,
            dual_values() returns a dictionary with the dual values for all linear
            constraints.

        Returns:
          The dual values associated to the best solution.

        Raises:
          ValueError: The best solution does not have an associated dual feasible
            solution.
          TypeError: Argument is not None, a LinearConstraint or an iterable of
            LinearConstraint.
          KeyError: LinearConstraint values requested for an invalid
            linear constraint (e.g. is not a LinearConstraint or is a linear
            constraint for another model).
        """
        if not self.has_dual_feasible_solution():
            raise ValueError(_NO_DUAL_SOLUTION_ERROR)
        assert self.solutions[0].dual_solution is not None
        if linear_constraints is None:
            return self.solutions[0].dual_solution.dual_values
        if isinstance(linear_constraints, model.LinearConstraint):
            return self.solutions[0].dual_solution.dual_values[linear_constraints]
        if isinstance(linear_constraints, Iterable):
            return [self.solutions[0].dual_solution.dual_values[c] for c in linear_constraints]
        raise TypeError(f'unsupported type in argument for dual_values: {type(linear_constraints).__name__!r}')

    @overload
    def reduced_costs(self, variables: None=...) -> Dict[model.Variable, float]:
        ...

    @overload
    def reduced_costs(self, variables: model.Variable) -> float:
        ...

    @overload
    def reduced_costs(self, variables: Iterable[model.Variable]) -> List[float]:
        ...

    def reduced_costs(self, variables=None):
        """The reduced costs associated to the best solution.

        If there is at least one primal feasible solution, this corresponds to the
        reduced costs associated to the best primal feasible solution. An error will
        be raised if the best solution does not have an associated dual feasible
        solution.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            reduced costs to return. If not provided, reduced_costs() returns a
            dictionary with the reduced costs for all variables.

        Returns:
          The reduced costs associated to the best solution.

        Raises:
          ValueError: The best solution does not have an associated dual feasible
            solution.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
        if not self.has_dual_feasible_solution():
            raise ValueError(_NO_DUAL_SOLUTION_ERROR)
        assert self.solutions[0].dual_solution is not None
        if variables is None:
            return self.solutions[0].dual_solution.reduced_costs
        if isinstance(variables, model.Variable):
            return self.solutions[0].dual_solution.reduced_costs[variables]
        if isinstance(variables, Iterable):
            return [self.solutions[0].dual_solution.reduced_costs[v] for v in variables]
        raise TypeError(f'unsupported type in argument for reduced_costs: {type(variables).__name__!r}')

    def has_dual_ray(self) -> bool:
        """Indicates if at least one dual ray is available.

        This is NOT guaranteed to be true when termination.reason is
        TerminationReason.Infeasible.

        Returns:
          True if at least one dual ray is available.
        """
        return bool(self.dual_rays)

    @overload
    def ray_dual_values(self, linear_constraints: None=...) -> Dict[model.LinearConstraint, float]:
        ...

    @overload
    def ray_dual_values(self, linear_constraints: model.LinearConstraint) -> float:
        ...

    @overload
    def ray_dual_values(self, linear_constraints: Iterable[model.LinearConstraint]) -> List[float]:
        ...

    def ray_dual_values(self, linear_constraints=None):
        """The dual values from the first dual ray.

        An error will be raised if there are no dual rays.

        Args:
          linear_constraints: an optional LinearConstraint or iterator of
            LinearConstraint indicating what dual values to return. If not provided,
            ray_dual_values() returns a dictionary with the dual values for all
            linear constraints.

        Returns:
          The dual values from the first dual ray.

        Raises:
          ValueError: There are no dual rays.
          TypeError: Argument is not None, a LinearConstraint or an iterable of
            LinearConstraint.
          KeyError: LinearConstraint values requested for an invalid
            linear constraint (e.g. is not a LinearConstraint or is a linear
            constraint for another model).
        """
        if not self.has_dual_ray():
            raise ValueError('No dual ray available.')
        if linear_constraints is None:
            return self.dual_rays[0].dual_values
        if isinstance(linear_constraints, model.LinearConstraint):
            return self.dual_rays[0].dual_values[linear_constraints]
        if isinstance(linear_constraints, Iterable):
            return [self.dual_rays[0].dual_values[v] for v in linear_constraints]
        raise TypeError(f'unsupported type in argument for ray_dual_values: {type(linear_constraints).__name__!r}')

    @overload
    def ray_reduced_costs(self, variables: None=...) -> Dict[model.Variable, float]:
        ...

    @overload
    def ray_reduced_costs(self, variables: model.Variable) -> float:
        ...

    @overload
    def ray_reduced_costs(self, variables: Iterable[model.Variable]) -> List[float]:
        ...

    def ray_reduced_costs(self, variables=None):
        """The reduced costs from the first dual ray.

        An error will be raised if there are no dual rays.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            reduced costs to return. If not provided, ray_reduced_costs() returns a
            dictionary with the reduced costs for all variables.

        Returns:
          The reduced costs from the first dual ray.

        Raises:
          ValueError: There are no dual rays.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
        if not self.has_dual_ray():
            raise ValueError('No dual ray available.')
        if variables is None:
            return self.dual_rays[0].reduced_costs
        if isinstance(variables, model.Variable):
            return self.dual_rays[0].reduced_costs[variables]
        if isinstance(variables, Iterable):
            return [self.dual_rays[0].reduced_costs[v] for v in variables]
        raise TypeError(f'unsupported type in argument for ray_reduced_costs: {type(variables).__name__!r}')

    def has_basis(self) -> bool:
        """Indicates if the best solution has an associated basis.

        This is NOT guaranteed to be true when termination.reason is
        TerminationReason.Optimal. It also may be true even when the best solution
        does not have an associated primal feasible solution.

        Returns:
          True if the best solution has an associated basis.
        """
        if not self.solutions:
            return False
        return self.solutions[0].basis is not None

    @overload
    def constraint_status(self, linear_constraints: None=...) -> Dict[model.LinearConstraint, solution.BasisStatus]:
        ...

    @overload
    def constraint_status(self, linear_constraints: model.LinearConstraint) -> solution.BasisStatus:
        ...

    @overload
    def constraint_status(self, linear_constraints: Iterable[model.LinearConstraint]) -> List[solution.BasisStatus]:
        ...

    def constraint_status(self, linear_constraints=None):
        """The constraint basis status associated to the best solution.

        If there is at least one primal feasible solution, this corresponds to the
        basis associated to the best primal feasible solution. An error will
        be raised if the best solution does not have an associated basis.


        Args:
          linear_constraints: an optional LinearConstraint or iterator of
            LinearConstraint indicating what constraint statuses to return. If not
            provided, returns a dictionary with the constraint statuses for all
            linear constraints.

        Returns:
          The constraint basis status associated to the best solution.

        Raises:
          ValueError: The best solution does not have an associated basis.
          TypeError: Argument is not None, a LinearConstraint or an iterable of
            LinearConstraint.
          KeyError: LinearConstraint values requested for an invalid
            linear constraint (e.g. is not a LinearConstraint or is a linear
            constraint for another model).
        """
        if not self.has_basis():
            raise ValueError(_NO_BASIS_ERROR)
        assert self.solutions[0].basis is not None
        if linear_constraints is None:
            return self.solutions[0].basis.constraint_status
        if isinstance(linear_constraints, model.LinearConstraint):
            return self.solutions[0].basis.constraint_status[linear_constraints]
        if isinstance(linear_constraints, Iterable):
            return [self.solutions[0].basis.constraint_status[c] for c in linear_constraints]
        raise TypeError(f'unsupported type in argument for constraint_status: {type(linear_constraints).__name__!r}')

    @overload
    def variable_status(self, variables: None=...) -> Dict[model.Variable, solution.BasisStatus]:
        ...

    @overload
    def variable_status(self, variables: model.Variable) -> solution.BasisStatus:
        ...

    @overload
    def variable_status(self, variables: Iterable[model.Variable]) -> List[solution.BasisStatus]:
        ...

    def variable_status(self, variables=None):
        """The variable basis status associated to the best solution.

        If there is at least one primal feasible solution, this corresponds to the
        basis associated to the best primal feasible solution. An error will
        be raised if the best solution does not have an associated basis.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            reduced costs to return. If not provided, variable_status() returns a
            dictionary with the reduced costs for all variables.

        Returns:
          The variable basis status associated to the best solution.

        Raises:
          ValueError: The best solution does not have an associated basis.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
        if not self.has_basis():
            raise ValueError(_NO_BASIS_ERROR)
        assert self.solutions[0].basis is not None
        if variables is None:
            return self.solutions[0].basis.variable_status
        if isinstance(variables, model.Variable):
            return self.solutions[0].basis.variable_status[variables]
        if isinstance(variables, Iterable):
            return [self.solutions[0].basis.variable_status[v] for v in variables]
        raise TypeError(f'unsupported type in argument for variable_status: {type(variables).__name__!r}')