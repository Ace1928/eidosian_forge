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
def _add_enforced_linear_constraint(self, helper: mbh.ModelBuilderHelper, var: Variable, value: bool, name: Optional[str]) -> 'EnforcedLinearConstraint':
    """Adds an enforced linear constraint to the model."""
    c = EnforcedLinearConstraint(helper)
    c.indicator_variable = var
    c.indicator_value = value
    flat_expr = _as_flat_linear_expression(self.__expr)
    helper.add_terms_to_enforced_constraint(c.index, flat_expr._variable_indices, flat_expr._coefficients)
    helper.set_enforced_constraint_lower_bound(c.index, self.__lb - flat_expr._offset)
    helper.set_enforced_constraint_upper_bound(c.index, self.__ub - flat_expr._offset)
    if name is not None:
        helper.set_enforced_constraint_name(c.index, name)
    return c