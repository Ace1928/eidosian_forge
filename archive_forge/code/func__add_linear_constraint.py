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
def _add_linear_constraint(self, helper: mbh.ModelBuilderHelper, name: Optional[str]) -> 'LinearConstraint':
    c = LinearConstraint(helper)
    flat_expr = _as_flat_linear_expression(self.__expr)
    helper.add_terms_to_constraint(c.index, flat_expr._variable_indices, flat_expr._coefficients)
    helper.set_constraint_lower_bound(c.index, self.__lb - flat_expr._offset)
    helper.set_constraint_upper_bound(c.index, self.__ub - flat_expr._offset)
    if name is not None:
        helper.set_constraint_name(c.index, name)
    return c