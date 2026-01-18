from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def _choose_recipe(argnum, idx, gradient_recipes, shifts, tape):
    """Obtain the gradient recipe for an indicated parameter from provided
    ``gradient_recipes``. If none is provided, use the recipe of the operation instead."""
    arg_idx = argnum.index(idx)
    recipe = gradient_recipes[arg_idx]
    if recipe is not None:
        recipe = process_shifts(np.array(recipe))
    else:
        op_shifts = None if shifts is None else shifts[arg_idx]
        recipe = _get_operation_recipe(tape, idx, shifts=op_shifts)
    return recipe