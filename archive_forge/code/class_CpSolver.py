import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
class CpSolver:
    """Main solver class.

    The purpose of this class is to search for a solution to the model provided
    to the solve() method.

    Once solve() is called, this class allows inspecting the solution found
    with the value() and boolean_value() methods, as well as general statistics
    about the solve procedure.
    """

    def __init__(self):
        self.__solution: Optional[cp_model_pb2.CpSolverResponse] = None
        self.parameters: sat_parameters_pb2.SatParameters = sat_parameters_pb2.SatParameters()
        self.log_callback: Optional[swig_helper.LogCallback] = None
        self.__solve_wrapper: Optional[swig_helper.SolveWrapper] = None
        self.__lock: threading.Lock = threading.Lock()

    def solve(self, model: CpModel, solution_callback: Optional['CpSolverSolutionCallback']=None) -> cp_model_pb2.CpSolverStatus:
        """Solves a problem and passes each solution to the callback if not null."""
        with self.__lock:
            self.__solve_wrapper = swig_helper.SolveWrapper()
        self.__solve_wrapper.set_parameters(self.parameters)
        if solution_callback is not None:
            self.__solve_wrapper.add_solution_callback(solution_callback)
        if self.log_callback is not None:
            self.__solve_wrapper.add_log_callback(self.log_callback)
        self.__solution = self.__solve_wrapper.solve(model.proto)
        if solution_callback is not None:
            self.__solve_wrapper.clear_solution_callback(solution_callback)
        with self.__lock:
            self.__solve_wrapper = None
        return self.__solution.status

    def stop_search(self) -> None:
        """Stops the current search asynchronously."""
        with self.__lock:
            if self.__solve_wrapper:
                self.__solve_wrapper.stop_search()

    def value(self, expression: LinearExprT) -> int:
        """Returns the value of a linear expression after solve."""
        return evaluate_linear_expr(expression, self._solution)

    def values(self, variables: _IndexOrSeries) -> pd.Series:
        """Returns the values of the input variables.

        If `variables` is a `pd.Index`, then the output will be indexed by the
        variables. If `variables` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          variables (Union[pd.Index, pd.Series]): The set of variables from which to
            get the values.

        Returns:
          pd.Series: The values of all variables in the set.
        """
        solution = self._solution
        return _attribute_series(func=lambda v: solution.solution[v.index], values=variables)

    def boolean_value(self, literal: LiteralT) -> bool:
        """Returns the boolean value of a literal after solve."""
        return evaluate_boolean_expression(literal, self._solution)

    def boolean_values(self, variables: _IndexOrSeries) -> pd.Series:
        """Returns the values of the input variables.

        If `variables` is a `pd.Index`, then the output will be indexed by the
        variables. If `variables` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          variables (Union[pd.Index, pd.Series]): The set of variables from which to
            get the values.

        Returns:
          pd.Series: The values of all variables in the set.
        """
        solution = self._solution
        return _attribute_series(func=lambda literal: evaluate_boolean_expression(literal, solution), values=variables)

    @property
    def objective_value(self) -> float:
        """Returns the value of the objective after solve."""
        return self._solution.objective_value

    @property
    def best_objective_bound(self) -> float:
        """Returns the best lower (upper) bound found when min(max)imizing."""
        return self._solution.best_objective_bound

    @property
    def num_booleans(self) -> int:
        """Returns the number of boolean variables managed by the SAT solver."""
        return self._solution.num_booleans

    @property
    def num_conflicts(self) -> int:
        """Returns the number of conflicts since the creation of the solver."""
        return self._solution.num_conflicts

    @property
    def num_branches(self) -> int:
        """Returns the number of search branches explored by the solver."""
        return self._solution.num_branches

    @property
    def wall_time(self) -> float:
        """Returns the wall time in seconds since the creation of the solver."""
        return self._solution.wall_time

    @property
    def user_time(self) -> float:
        """Returns the user time in seconds since the creation of the solver."""
        return self._solution.user_time

    @property
    def response_proto(self) -> cp_model_pb2.CpSolverResponse:
        """Returns the response object."""
        return self._solution

    def response_stats(self) -> str:
        """Returns some statistics on the solution found as a string."""
        return swig_helper.CpSatHelper.solver_response_stats(self._solution)

    def sufficient_assumptions_for_infeasibility(self) -> Sequence[int]:
        """Returns the indices of the infeasible assumptions."""
        return self._solution.sufficient_assumptions_for_infeasibility

    def status_name(self, status: Optional[Any]=None) -> str:
        """Returns the name of the status returned by solve()."""
        if status is None:
            status = self._solution.status
        return cp_model_pb2.CpSolverStatus.Name(status)

    def solution_info(self) -> str:
        """Returns some information on the solve process.

        Returns some information on how the solution was found, or the reason
        why the model or the parameters are invalid.

        Raises:
          RuntimeError: if solve() has not been called.
        """
        return self._solution.solution_info

    @property
    def _solution(self) -> cp_model_pb2.CpSolverResponse:
        """Checks solve() has been called, and returns the solution."""
        if self.__solution is None:
            raise RuntimeError('solve() has not been called.')
        return self.__solution

    def BestObjectiveBound(self) -> float:
        return self.best_objective_bound

    def BooleanValue(self, literal: LiteralT) -> bool:
        return self.boolean_value(literal)

    def BooleanValues(self, variables: _IndexOrSeries) -> pd.Series:
        return self.boolean_values(variables)

    def NumBooleans(self) -> int:
        return self.num_booleans

    def NumConflicts(self) -> int:
        return self.num_conflicts

    def NumBranches(self) -> int:
        return self.num_branches

    def ObjectiveValue(self) -> float:
        return self.objective_value

    def ResponseProto(self) -> cp_model_pb2.CpSolverResponse:
        return self.response_proto

    def ResponseStats(self) -> str:
        return self.response_stats()

    def Solve(self, model: CpModel, solution_callback: Optional['CpSolverSolutionCallback']=None) -> cp_model_pb2.CpSolverStatus:
        return self.solve(model, solution_callback)

    def SolutionInfo(self) -> str:
        return self.solution_info()

    def StatusName(self, status: Optional[Any]=None) -> str:
        return self.status_name(status)

    def StopSearch(self) -> None:
        self.stop_search()

    def SufficientAssumptionsForInfeasibility(self) -> Sequence[int]:
        return self.sufficient_assumptions_for_infeasibility()

    def UserTime(self) -> float:
        return self.user_time

    def Value(self, expression: LinearExprT) -> int:
        return self.value(expression)

    def Values(self, variables: _IndexOrSeries) -> pd.Series:
        return self.values(variables)

    def WallTime(self) -> float:
        return self.wall_time

    def SolveWithSolutionCallback(self, model: CpModel, callback: 'CpSolverSolutionCallback') -> cp_model_pb2.CpSolverStatus:
        """DEPRECATED Use solve() with the callback argument."""
        warnings.warn('solve_with_solution_callback is deprecated; use solve() with' + 'the callback argument.', DeprecationWarning)
        return self.solve(model, callback)

    def SearchForAllSolutions(self, model: CpModel, callback: 'CpSolverSolutionCallback') -> cp_model_pb2.CpSolverStatus:
        """DEPRECATED Use solve() with the right parameter.

        Search for all solutions of a satisfiability problem.

        This method searches for all feasible solutions of a given model.
        Then it feeds the solution to the callback.

        Note that the model cannot contain an objective.

        Args:
          model: The model to solve.
          callback: The callback that will be called at each solution.

        Returns:
          The status of the solve:

          * *FEASIBLE* if some solutions have been found
          * *INFEASIBLE* if the solver has proved there are no solution
          * *OPTIMAL* if all solutions have been found
        """
        warnings.warn('search_for_all_solutions is deprecated; use solve() with' + 'enumerate_all_solutions = True.', DeprecationWarning)
        if model.has_objective():
            raise TypeError('Search for all solutions is only defined on satisfiability problems')
        enumerate_all = self.parameters.enumerate_all_solutions
        self.parameters.enumerate_all_solutions = True
        self.solve(model, callback)
        self.parameters.enumerate_all_solutions = enumerate_all
        return self.__solution.status