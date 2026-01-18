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
def _sum_as_flat_linear_expression(to_process: List[Tuple[LinearExprT, float]], offset: float=0.0) -> _LinearExpression:
    """Creates a _LinearExpression as the sum of terms."""
    indices = []
    coeffs = []
    helper = None
    while to_process:
        expr, coeff = to_process.pop()
        if isinstance(expr, _Sum):
            to_process.append((expr._left, coeff))
            to_process.append((expr._right, coeff))
        elif isinstance(expr, Variable):
            indices.append([expr.index])
            coeffs.append([coeff])
            if helper is None:
                helper = expr.helper
        elif mbn.is_a_number(expr):
            offset += coeff * cast(NumberT, expr)
        elif isinstance(expr, _Product):
            to_process.append((expr._expression, coeff * expr._coefficient))
        elif isinstance(expr, _LinearExpression):
            offset += coeff * expr._offset
            if expr._helper is not None:
                indices.append(expr.variable_indices)
                coeffs.append(np.multiply(expr.coefficients, coeff))
                if helper is None:
                    helper = expr._helper
        else:
            raise TypeError('Unrecognized linear expression: ' + str(expr) + f' {type(expr)}')
    if helper is not None:
        all_indices: npt.NDArray[np.int32] = np.concatenate(indices, axis=0)
        all_coeffs: npt.NDArray[np.double] = np.concatenate(coeffs, axis=0)
        sorted_indices, sorted_coefficients = helper.sort_and_regroup_terms(all_indices, all_coeffs)
        return _LinearExpression(sorted_indices, sorted_coefficients, offset, helper)
    else:
        assert not indices
        assert not coeffs
        return _LinearExpression(_variable_indices=np.zeros(dtype=np.int32, shape=[0]), _coefficients=np.zeros(dtype=np.double, shape=[0]), _offset=offset, _helper=None)