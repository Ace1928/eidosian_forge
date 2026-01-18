import itertools as it
import warnings
from functools import partial
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP, StateMP, VarianceMP
from pennylane.transforms import transform
from .general_shift_rules import (
from .gradient_transform import find_and_validate_gradient_methods
from .parameter_shift import _get_operation_recipe
from .hessian_transform import _process_jacs
def _collect_recipes(tape, argnum, method_map, diagonal_shifts, off_diagonal_shifts):
    """Extract second order recipes for the tape operations for the diagonal of the Hessian
    as well as the first-order derivative recipes for the off-diagonal entries.
    """
    diag_argnum = qml.math.diag(argnum)
    offdiag_argnum = qml.math.any(argnum ^ qml.math.diag(qml.math.diag(argnum)), axis=0)
    diag_recipes = []
    partial_offdiag_recipes = []
    diag_shifts_idx = offdiag_shifts_idx = 0
    for i, (d, od) in enumerate(zip(diag_argnum, offdiag_argnum)):
        if not d or method_map[i] == '0':
            diag_recipes.append(None)
        else:
            diag_shifts = None if diagonal_shifts is None else diagonal_shifts[diag_shifts_idx]
            diag_recipes.append(_get_operation_recipe(tape, i, diag_shifts, order=2))
            diag_shifts_idx += 1
        if not od or method_map[i] == '0':
            partial_offdiag_recipes.append((None, None, None))
        else:
            offdiag_shifts = None if off_diagonal_shifts is None else off_diagonal_shifts[offdiag_shifts_idx]
            partial_offdiag_recipes.append(_get_operation_recipe(tape, i, offdiag_shifts, order=1))
            offdiag_shifts_idx += 1
    return (diag_recipes, partial_offdiag_recipes)