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
@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class VarEqVar(_BoundedLinearExpr):
    """Represents var == var."""
    __slots__ = ('left', 'right')
    left: Variable
    right: Variable

    def __str__(self):
        return f'{self.left} == {self.right}'

    def __repr__(self):
        return self.__str__()

    def __bool__(self) -> bool:
        return hash(self.left) == hash(self.right)

    def _add_linear_constraint(self, helper: mbh.ModelBuilderHelper, name: str) -> 'LinearConstraint':
        c = LinearConstraint(helper)
        helper.set_constraint_lower_bound(c.index, 0.0)
        helper.set_constraint_upper_bound(c.index, 0.0)
        helper.add_term_to_constraint(c.index, self.left.index, 1.0)
        helper.add_term_to_constraint(c.index, self.right.index, -1.0)
        helper.set_constraint_name(c.index, name)
        return c

    def _add_enforced_linear_constraint(self, helper: mbh.ModelBuilderHelper, var: Variable, value: bool, name: str) -> 'EnforcedLinearConstraint':
        """Adds an enforced linear constraint to the model."""
        c = EnforcedLinearConstraint(helper)
        c.indicator_variable = var
        c.indicator_value = value
        helper.set_enforced_constraint_lower_bound(c.index, 0.0)
        helper.set_enforced_constraint_upper_bound(c.index, 0.0)
        helper.add_term_to_enforced_constraint(c.index, self.left.index, 1.0)
        helper.add_term_to_enforced_constraint(c.index, self.right.index, -1.0)
        helper.set_enforced_constraint_name(c.index, name)
        return c