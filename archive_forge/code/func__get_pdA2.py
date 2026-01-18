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
def _get_pdA2(results, tape, pdA2_fn, non_involutory_indices, var_indices):
    """The main auxiliary function to get the partial derivative of <A^2>."""
    pdA2 = 0
    if non_involutory_indices:
        pdA2 = pdA2_fn(results)
        if (involutory := (set(var_indices) - set(non_involutory_indices))):
            if tape.shots.has_partitioned_shots:
                pdA2 = tuple((_put_zeros_in_pdA2_involutory(tape, pdA2_shot_comp, involutory) for pdA2_shot_comp in pdA2))
            else:
                pdA2 = _put_zeros_in_pdA2_involutory(tape, pdA2, involutory)
    return pdA2