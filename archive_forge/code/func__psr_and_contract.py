from typing import Sequence, Callable
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution, HardwareHamiltonian
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from .gradient_transform import (
def _psr_and_contract(res_list, cjacs, int_prefactor):
    """Execute the parameter-shift rule and contract with classical Jacobians.
            This function assumes a single generating term for the pulse parameter
            of interest"""
    res = jnp.stack(res_list)
    if use_broadcasting:
        res = res[:, 1:-1]
    else:
        shape = jnp.shape(res)
        new_shape = (shape[0] // num_shifts, num_shifts) + shape[1:]
        res = jnp.moveaxis(jnp.reshape(res, new_shape), 1, 0)
    return _contract(psr_coeffs, res, cjacs) * int_prefactor