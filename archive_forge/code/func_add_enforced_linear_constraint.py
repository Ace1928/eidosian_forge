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
def add_enforced_linear_constraint(self, linear_expr: LinearExprT, ivar: 'Variable', ivalue: bool, lb: NumberT=-math.inf, ub: NumberT=math.inf, name: Optional[str]=None) -> EnforcedLinearConstraint:
    """Adds the constraint: `ivar == ivalue => lb <= linear_expr <= ub` with the given name."""
    ct = EnforcedLinearConstraint(self.__helper)
    ct.indicator_variable = ivar
    ct.indicator_value = ivalue
    if name:
        self.__helper.set_constraint_name(ct.index, name)
    if mbn.is_a_number(linear_expr):
        self.__helper.set_constraint_lower_bound(ct.index, lb - linear_expr)
        self.__helper.set_constraint_upper_bound(ct.index, ub - linear_expr)
    elif isinstance(linear_expr, Variable):
        self.__helper.set_constraint_lower_bound(ct.index, lb)
        self.__helper.set_constraint_upper_bound(ct.index, ub)
        self.__helper.add_term_to_constraint(ct.index, linear_expr.index, 1.0)
    elif isinstance(linear_expr, LinearExpr):
        flat_expr = _as_flat_linear_expression(linear_expr)
        self.__helper.set_constraint_lower_bound(ct.index, lb - flat_expr._offset)
        self.__helper.set_constraint_upper_bound(ct.index, ub - flat_expr._offset)
        self.__helper.add_terms_to_constraint(ct.index, flat_expr._variable_indices, flat_expr._coefficients)
    else:
        raise TypeError(f'Not supported: Model.add_enforced_linear_constraint({linear_expr}) with type {type(linear_expr)}')
    return ct