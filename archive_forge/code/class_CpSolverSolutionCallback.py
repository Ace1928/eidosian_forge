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
class CpSolverSolutionCallback(swig_helper.SolutionCallback):
    """Solution callback.

    This class implements a callback that will be called at each new solution
    found during search.

    The method on_solution_callback() will be called by the solver, and must be
    implemented. The current solution can be queried using the boolean_value()
    and value() methods.

    These methods returns the same information as their counterpart in the
    `CpSolver` class.
    """

    def __init__(self):
        swig_helper.SolutionCallback.__init__(self)

    def OnSolutionCallback(self) -> None:
        """Proxy for the same method in snake case."""
        self.on_solution_callback()

    def boolean_value(self, lit: LiteralT) -> bool:
        """Returns the boolean value of a boolean literal.

        Args:
            lit: A boolean variable or its negation.

        Returns:
            The Boolean value of the literal in the solution.

        Raises:
            RuntimeError: if `lit` is not a boolean variable or its negation.
        """
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        if isinstance(lit, numbers.Integral):
            return bool(lit)
        if isinstance(lit, IntVar) or isinstance(lit, _NotBooleanVariable):
            return self.SolutionBooleanValue(cast(Union[IntVar, _NotBooleanVariable], lit).index)
        if cmh.is_boolean(lit):
            return bool(lit)
        raise TypeError(f'Cannot interpret {lit} as a boolean expression.')

    def value(self, expression: LinearExprT) -> int:
        """Evaluates an linear expression in the current solution.

        Args:
            expression: a linear expression of the model.

        Returns:
            An integer value equal to the evaluation of the linear expression
            against the current solution.

        Raises:
            RuntimeError: if 'expression' is not a LinearExpr.
        """
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        value = 0
        to_process = [(expression, 1)]
        while to_process:
            expr, coeff = to_process.pop()
            if isinstance(expr, numbers.Integral):
                value += int(expr) * coeff
            elif isinstance(expr, _ProductCst):
                to_process.append((expr.expression(), coeff * expr.coefficient()))
            elif isinstance(expr, _Sum):
                to_process.append((expr.left(), coeff))
                to_process.append((expr.right(), coeff))
            elif isinstance(expr, _SumArray):
                for e in expr.expressions():
                    to_process.append((e, coeff))
                    value += expr.constant() * coeff
            elif isinstance(expr, _WeightedSum):
                for e, c in zip(expr.expressions(), expr.coefficients()):
                    to_process.append((e, coeff * c))
                value += expr.constant() * coeff
            elif isinstance(expr, IntVar):
                value += coeff * self.SolutionIntegerValue(expr.index)
            elif isinstance(expr, _NotBooleanVariable):
                value += coeff * (1 - self.SolutionIntegerValue(expr.negated().index))
            else:
                raise TypeError(f'cannot interpret {expression} as a linear expression.')
        return value

    def has_response(self) -> bool:
        return self.HasResponse()

    def stop_search(self) -> None:
        """Stops the current search asynchronously."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        self.StopSearch()

    @property
    def objective_value(self) -> float:
        """Returns the value of the objective after solve."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.ObjectiveValue()

    @property
    def best_objective_bound(self) -> float:
        """Returns the best lower (upper) bound found when min(max)imizing."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.BestObjectiveBound()

    @property
    def num_booleans(self) -> int:
        """Returns the number of boolean variables managed by the SAT solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.NumBooleans()

    @property
    def num_conflicts(self) -> int:
        """Returns the number of conflicts since the creation of the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.NumConflicts()

    @property
    def num_branches(self) -> int:
        """Returns the number of search branches explored by the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.NumBranches()

    @property
    def num_integer_propagations(self) -> int:
        """Returns the number of integer propagations done by the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.NumIntegerPropagations()

    @property
    def num_boolean_propagations(self) -> int:
        """Returns the number of Boolean propagations done by the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.NumBooleanPropagations()

    @property
    def deterministic_time(self) -> float:
        """Returns the determistic time in seconds since the creation of the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.DeterministicTime()

    @property
    def wall_time(self) -> float:
        """Returns the wall time in seconds since the creation of the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.WallTime()

    @property
    def user_time(self) -> float:
        """Returns the user time in seconds since the creation of the solver."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.UserTime()

    @property
    def response_proto(self) -> cp_model_pb2.CpSolverResponse:
        """Returns the response object."""
        if not self.has_response():
            raise RuntimeError('solve() has not been called.')
        return self.Response()
    Value = value
    BooleanValue = boolean_value