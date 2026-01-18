from typing import Callable, Sequence
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_strings, _pauli_decompose
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .pulse_gradient import _assert_has_jax, raise_pulse_diff_on_qnode
from .gradient_transform import (
def _parshift_and_contract_single(res_list, coeffs):
    """Execute the standard parameter-shift rule on a list of results
        and contract with Pauli basis coefficients."""
    psr_deriv = ((res := qml.math.stack(res_list))[::2] - res[1::2]) / 2
    return qml.math.tensordot(psr_deriv, coeffs, axes=[[0], [0]])