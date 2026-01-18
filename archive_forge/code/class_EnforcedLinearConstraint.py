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
class EnforcedLinearConstraint:
    """Stores an enforced linear equation, also name indicator constraint.

    Example:
        x = model.new_num_var(0, 10, 'x')
        y = model.new_num_var(0, 10, 'y')
        z = model.new_bool_var('z')

        enforced_linear_constraint = model.add_enforced(x + 2 * y == 5, z, False)
    """

    def __init__(self, helper: mbh.ModelBuilderHelper, index: Optional[IntegerT]=None):
        if index is None:
            self.__index = helper.add_enforced_linear_constraint()
        else:
            if not helper.is_enforced_linear_constraint(index):
                raise ValueError(f'the given index {index} does not refer to an enforced linear constraint')
            self.__index = index
        self.__helper: mbh.ModelBuilderHelper = helper

    @property
    def index(self) -> IntegerT:
        """Returns the index of the constraint in the helper."""
        return self.__index

    @property
    def helper(self) -> mbh.ModelBuilderHelper:
        """Returns the ModelBuilderHelper instance."""
        return self.__helper

    @property
    def lower_bound(self) -> np.double:
        return self.__helper.enforced_constraint_lower_bound(self.__index)

    @lower_bound.setter
    def lower_bound(self, bound: NumberT) -> None:
        self.__helper.set_enforced_constraint_lower_bound(self.__index, bound)

    @property
    def upper_bound(self) -> np.double:
        return self.__helper.enforced_constraint_upper_bound(self.__index)

    @upper_bound.setter
    def upper_bound(self, bound: NumberT) -> None:
        self.__helper.set_enforced_constraint_upper_bound(self.__index, bound)

    @property
    def indicator_variable(self) -> 'Variable':
        enforcement_var_index = self.__helper.enforced_constraint_indicator_variable_index(self.__index)
        return Variable(self.__helper, enforcement_var_index, None, None, None)

    @indicator_variable.setter
    def indicator_variable(self, var: 'Variable') -> None:
        self.__helper.set_enforced_constraint_indicator_variable_index(self.__index, var.index)

    @property
    def indicator_value(self) -> bool:
        return self.__helper.enforced_constraint_indicator_value(self.__index)

    @indicator_value.setter
    def indicator_value(self, value: bool) -> None:
        self.__helper.set_enforced_constraint_indicator_value(self.__index, value)

    @property
    def name(self) -> str:
        constraint_name = self.__helper.enforced_constraint_name(self.__index)
        if constraint_name:
            return constraint_name
        return f'enforced_linear_constraint#{self.__index}'

    @name.setter
    def name(self, name: str) -> None:
        return self.__helper.set_enforced_constraint_name(self.__index, name)

    def is_always_false(self) -> bool:
        """Returns True if the constraint is always false.

        Usually, it means that it was created by model.add(False)
        """
        return self.lower_bound > self.upper_bound

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'EnforcedLinearConstraint({self.name}, lb={self.lower_bound}, ub={self.upper_bound}, var_indices={self.helper.enforced_constraint_var_indices(self.index)}, coefficients={self.helper.enforced_constraint_coefficients(self.index)}, indicator_variable={self.indicator_variable} indicator_value={self.indicator_value})'

    def set_coefficient(self, var: Variable, coeff: NumberT) -> None:
        """Sets the coefficient of the variable in the constraint."""
        if self.is_always_false():
            raise ValueError(f'Constraint {self.index} is always false and cannot be modified')
        self.__helper.set_enforced_constraint_coefficient(self.__index, var.index, coeff)

    def add_term(self, var: Variable, coeff: NumberT) -> None:
        """Adds var * coeff to the constraint."""
        if self.is_always_false():
            raise ValueError(f'Constraint {self.index} is always false and cannot be modified')
        self.__helper.safe_add_term_to_enforced_constraint(self.__index, var.index, coeff)

    def clear_terms(self) -> None:
        """Clear all terms of the constraint."""
        self.__helper.clear_enforced_constraint_terms(self.__index)