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
def _add_linear_constraint_to_helper(bounded_expr: Union[bool, _BoundedLinearExpr], helper: mbh.ModelBuilderHelper, name: Optional[str]):
    """Creates a new linear constraint in the helper.

    It handles boolean values (which might arise in the construction of
    BoundedLinearExpressions).

    Args:
      bounded_expr: The bounded expression used to create the constraint.
      helper: The helper to create the constraint.
      name: The name of the constraint to be created.

    Returns:
      LinearConstraint: a constraint in the helper corresponding to the input.

    Raises:
      TypeError: If constraint is an invalid type.
    """
    if isinstance(bounded_expr, bool):
        c = LinearConstraint(helper)
        if name is not None:
            helper.set_constraint_name(c.index, name)
        if bounded_expr:
            helper.set_constraint_lower_bound(c.index, 0.0)
            helper.set_constraint_upper_bound(c.index, 0.0)
        else:
            helper.set_constraint_lower_bound(c.index, 1)
            helper.set_constraint_upper_bound(c.index, -1)
        return c
    if isinstance(bounded_expr, _BoundedLinearExpr):
        return bounded_expr._add_linear_constraint(helper, name)
    raise TypeError('invalid type={}'.format(type(bounded_expr)))