import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
class Objective:
    """The objective for an optimization model.

    An objective is either of the form:
      min o + sum_{i in I} c_i * x_i + sum_{i, j in I, i <= j} q_i,j * x_i * x_j
    or
      max o + sum_{i in I} c_i * x_i + sum_{(i, j) in Q} q_i,j * x_i * x_j
    where x_i are the decision variables of the problem and where all pairs (i, j)
    in Q satisfy i <= j. The values of o, c_i and q_i,j should be finite and not
    NaN.

    The objective can be configured as follows:
      * offset: a float property, o above. Should be finite and not NaN.
      * is_maximize: a bool property, if the objective is to maximize or minimize.
      * set_linear_coefficient and get_linear_coefficient control the c_i * x_i
        terms. The variables must be from the same model as this objective, and
        the c_i must be finite and not NaN. The coefficient for any variable not
        set is 0.0, and setting a coefficient to 0.0 removes it from I above.
      * set_quadratic_coefficient and get_quadratic_coefficient control the
        q_i,j * x_i * x_j terms. The variables must be from the same model as this
        objective, and the q_i,j must be finite and not NaN. The coefficient for
        any pair of variables not set is 0.0, and setting a coefficient to 0.0
        removes the associated (i,j) from Q above.

    Every Objective is associated with a Model (defined below). Note that data
    describing the objective (e.g. offset) is owned by Model.storage, this class
    is simply a reference to that data. Do not create an Objective directly,
    use Model.objective to access the objective instead.

    The objective will be linear if only linear coefficients are set. This can be
    useful to avoid solve-time errors with solvers that do not accept quadratic
    objectives. To facilitate this linear objective guarantee we provide three
    functions to add to the objective:
      * add(), which accepts linear or quadratic expressions,
      * add_quadratic(), which also accepts linear or quadratic expressions and
        can be used to signal a quadratic objective is possible, and
      * add_linear(), which only accepts linear expressions and can be used to
        guarantee the objective remains linear.
    """
    __slots__ = ('_model',)

    def __init__(self, model: 'Model') -> None:
        """Do no invoke directly, use Model.objective."""
        self._model: 'Model' = model

    @property
    def is_maximize(self) -> bool:
        return self.model.storage.get_is_maximize()

    @is_maximize.setter
    def is_maximize(self, is_maximize: bool) -> None:
        self.model.storage.set_is_maximize(is_maximize)

    @property
    def offset(self) -> float:
        return self.model.storage.get_objective_offset()

    @offset.setter
    def offset(self, value: float) -> None:
        self.model.storage.set_objective_offset(value)

    @property
    def model(self) -> 'Model':
        return self._model

    def set_linear_coefficient(self, variable: Variable, coef: float) -> None:
        self.model.check_compatible(variable)
        self.model.storage.set_linear_objective_coefficient(variable.id, coef)

    def get_linear_coefficient(self, variable: Variable) -> float:
        self.model.check_compatible(variable)
        return self.model.storage.get_linear_objective_coefficient(variable.id)

    def linear_terms(self) -> Iterator[LinearTerm]:
        """Yields variable coefficient pairs for variables with nonzero objective coefficient in undefined order."""
        yield from self.model.linear_objective_terms()

    def add(self, objective: QuadraticTypes) -> None:
        """Adds the provided expression `objective` to the objective function.

        To ensure the objective remains linear through type checks, use
        add_linear().

        Args:
          objective: the expression to add to the objective function.
        """
        if isinstance(objective, (LinearBase, int, float)):
            self.add_linear(objective)
        elif isinstance(objective, QuadraticBase):
            self.add_quadratic(objective)
        else:
            raise TypeError(f'unsupported type in objective argument for Objective.add(): {type(objective).__name__!r}')

    def add_linear(self, objective: LinearTypes) -> None:
        """Adds the provided linear expression `objective` to the objective function."""
        if not isinstance(objective, (LinearBase, int, float)):
            raise TypeError(f'unsupported type in objective argument for Objective.add_linear(): {type(objective).__name__!r}')
        objective_expr = as_flat_linear_expression(objective)
        self.offset += objective_expr.offset
        for var, coefficient in objective_expr.terms.items():
            self.set_linear_coefficient(var, self.get_linear_coefficient(var) + coefficient)

    def add_quadratic(self, objective: QuadraticTypes) -> None:
        """Adds the provided quadratic expression `objective` to the objective function."""
        if not isinstance(objective, (QuadraticBase, LinearBase, int, float)):
            raise TypeError(f'unsupported type in objective argument for Objective.add(): {type(objective).__name__!r}')
        objective_expr = as_flat_quadratic_expression(objective)
        self.offset += objective_expr.offset
        for var, coefficient in objective_expr.linear_terms.items():
            self.set_linear_coefficient(var, self.get_linear_coefficient(var) + coefficient)
        for key, coefficient in objective_expr.quadratic_terms.items():
            self.set_quadratic_coefficient(key.first_var, key.second_var, self.get_quadratic_coefficient(key.first_var, key.second_var) + coefficient)

    def set_quadratic_coefficient(self, first_variable: Variable, second_variable: Variable, coef: float) -> None:
        self.model.check_compatible(first_variable)
        self.model.check_compatible(second_variable)
        self.model.storage.set_quadratic_objective_coefficient(first_variable.id, second_variable.id, coef)

    def get_quadratic_coefficient(self, first_variable: Variable, second_variable: Variable) -> float:
        self.model.check_compatible(first_variable)
        self.model.check_compatible(second_variable)
        return self.model.storage.get_quadratic_objective_coefficient(first_variable.id, second_variable.id)

    def quadratic_terms(self) -> Iterator[QuadraticTerm]:
        """Yields quadratic terms with nonzero objective coefficient in undefined order."""
        yield from self.model.quadratic_objective_terms()

    def as_linear_expression(self) -> LinearExpression:
        if any(self.quadratic_terms()):
            raise TypeError('Cannot get a quadratic objective as a linear expression')
        return as_flat_linear_expression(self.offset + LinearSum(self.linear_terms()))

    def as_quadratic_expression(self) -> QuadraticExpression:
        return as_flat_quadratic_expression(self.offset + LinearSum(self.linear_terms()) + QuadraticSum(self.quadratic_terms()))

    def clear(self) -> None:
        """Clears objective coefficients and offset. Does not change direction."""
        self.model.storage.clear_objective()