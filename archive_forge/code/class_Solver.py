import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
class Solver:
    """Main solver class.

    The purpose of this class is to search for a solution to the model provided
    to the solve() method.

    Once solve() is called, this class allows inspecting the solution found
    with the value() method, as well as general statistics about the solve
    procedure.
    """

    def __init__(self, solver_name: str):
        self.__solve_helper: mbh.ModelSolverHelper = mbh.ModelSolverHelper(solver_name)
        self.log_callback: Optional[Callable[[str], None]] = None

    def solver_is_supported(self) -> bool:
        """Checks whether the requested solver backend was found."""
        return self.__solve_helper.solver_is_supported()

    def set_time_limit_in_seconds(self, limit: NumberT) -> None:
        """Sets a time limit for the solve() call."""
        self.__solve_helper.set_time_limit_in_seconds(limit)

    def set_solver_specific_parameters(self, parameters: str) -> None:
        """Sets parameters specific to the solver backend."""
        self.__solve_helper.set_solver_specific_parameters(parameters)

    def enable_output(self, enabled: bool) -> None:
        """Controls the solver backend logs."""
        self.__solve_helper.enable_output(enabled)

    def solve(self, model: Model) -> SolveStatus:
        """Solves a problem and passes each solution to the callback if not null."""
        if self.log_callback is not None:
            self.__solve_helper.set_log_callback(self.log_callback)
        else:
            self.__solve_helper.clear_log_callback()
        self.__solve_helper.solve(model.helper)
        return SolveStatus(self.__solve_helper.status())

    def stop_search(self):
        """Stops the current search asynchronously."""
        self.__solve_helper.interrupt_solve()

    def value(self, expr: LinearExprT) -> np.double:
        """Returns the value of a linear expression after solve."""
        if not self.__solve_helper.has_solution():
            return pd.NA
        if mbn.is_a_number(expr):
            return expr
        elif isinstance(expr, Variable):
            return self.__solve_helper.var_value(expr.index)
        elif isinstance(expr, LinearExpr):
            flat_expr = _as_flat_linear_expression(expr)
            return self.__solve_helper.expression_value(flat_expr._variable_indices, flat_expr._coefficients, flat_expr._offset)
        else:
            raise TypeError(f'Unknown expression {expr!r} of type {type(expr)}')

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
        if not self.__solve_helper.has_solution():
            return _attribute_series(func=lambda v: pd.NA, values=variables)
        return _attribute_series(func=lambda v: self.__solve_helper.var_value(v.index), values=variables)

    def reduced_costs(self, variables: _IndexOrSeries) -> pd.Series:
        """Returns the reduced cost of the input variables.

        If `variables` is a `pd.Index`, then the output will be indexed by the
        variables. If `variables` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          variables (Union[pd.Index, pd.Series]): The set of variables from which to
            get the values.

        Returns:
          pd.Series: The reduced cost of all variables in the set.
        """
        if not self.__solve_helper.has_solution():
            return _attribute_series(func=lambda v: pd.NA, values=variables)
        return _attribute_series(func=lambda v: self.__solve_helper.reduced_cost(v.index), values=variables)

    def reduced_cost(self, var: Variable) -> np.double:
        """Returns the reduced cost of a linear expression after solve."""
        if not self.__solve_helper.has_solution():
            return pd.NA
        return self.__solve_helper.reduced_cost(var.index)

    def dual_values(self, constraints: _IndexOrSeries) -> pd.Series:
        """Returns the dual values of the input constraints.

        If `constraints` is a `pd.Index`, then the output will be indexed by the
        constraints. If `constraints` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          constraints (Union[pd.Index, pd.Series]): The set of constraints from
            which to get the dual values.

        Returns:
          pd.Series: The dual_values of all constraints in the set.
        """
        if not self.__solve_helper.has_solution():
            return _attribute_series(func=lambda v: pd.NA, values=constraints)
        return _attribute_series(func=lambda v: self.__solve_helper.dual_value(v.index), values=constraints)

    def dual_value(self, ct: LinearConstraint) -> np.double:
        """Returns the dual value of a linear constraint after solve."""
        if not self.__solve_helper.has_solution():
            return pd.NA
        return self.__solve_helper.dual_value(ct.index)

    def activity(self, ct: LinearConstraint) -> np.double:
        """Returns the activity of a linear constraint after solve."""
        if not self.__solve_helper.has_solution():
            return pd.NA
        return self.__solve_helper.activity(ct.index)

    @property
    def objective_value(self) -> np.double:
        """Returns the value of the objective after solve."""
        if not self.__solve_helper.has_solution():
            return pd.NA
        return self.__solve_helper.objective_value()

    @property
    def best_objective_bound(self) -> np.double:
        """Returns the best lower (upper) bound found when min(max)imizing."""
        if not self.__solve_helper.has_solution():
            return pd.NA
        return self.__solve_helper.best_objective_bound()

    @property
    def status_string(self) -> str:
        """Returns additional information of the last solve.

        It can describe why the model is invalid.
        """
        return self.__solve_helper.status_string()

    @property
    def wall_time(self) -> np.double:
        return self.__solve_helper.wall_time()

    @property
    def user_time(self) -> np.double:
        return self.__solve_helper.user_time()