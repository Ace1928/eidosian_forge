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
def _process_op_recipe(op, p_idx, order):
    """Process an existing recipe of an operation."""
    recipe = op.grad_recipe[p_idx]
    if recipe is None:
        return None
    recipe = qml.math.array(recipe)
    if order == 1:
        return process_shifts(recipe, batch_duplicates=False)
    try:
        period = frequencies_to_period(op.parameter_frequencies[p_idx])
    except qml.operation.ParameterFrequenciesUndefinedError:
        period = None
    if qml.math.allclose(recipe[:, 1], qml.math.ones_like(recipe[:, 1])):
        iter_c, iter_s = process_shifts(_iterate_shift_rule(recipe[:, ::2], order, period)).T
        return qml.math.stack([iter_c, qml.math.ones_like(iter_c), iter_s]).T
    return process_shifts(_iterate_shift_rule(recipe, order, period))