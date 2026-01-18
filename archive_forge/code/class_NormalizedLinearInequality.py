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
@dataclasses.dataclass
class NormalizedLinearInequality:
    """Represents an inequality lb <= expr <= ub where expr's offset is zero.

    The inequality is of the form:
        lb <= sum_{x in V} coefficients[x] * x <= ub
    where V is the set of keys of coefficients.
    """
    lb: float
    ub: float
    coefficients: Mapping[Variable, float]

    def __init__(self, *, lb: float, ub: float, expr: LinearTypes) -> None:
        """Raises a ValueError if expr's offset is infinite."""
        flat_expr = as_flat_linear_expression(expr)
        if math.isinf(flat_expr.offset):
            raise ValueError('Trying to create a linear constraint whose expression has an infinite offset.')
        self.lb = lb - flat_expr.offset
        self.ub = ub - flat_expr.offset
        self.coefficients = flat_expr.terms