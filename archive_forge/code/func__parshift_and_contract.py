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
def _parshift_and_contract(results, coeffs, single_measure, single_shot_entry):
    """Compute parameter-shift tape derivatives and contract them with coefficients.

    Args:
        results (list[tensor_like]): Tape execution results to be processed.
        coeffs (list[tensor_like]): Coefficients to be contracted.
        single_measure (bool): whether the tape execution results contain single measurements.
        single_shot_entry (bool): whether the tape execution results were obtained with a single
            shots setting.

    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: contraction between the
        parameter-shift derivative computed from ``results`` and the ``coeffs``.
    """

    def _parshift_and_contract_single(res_list, coeffs):
        """Execute the standard parameter-shift rule on a list of results
        and contract with Pauli basis coefficients."""
        psr_deriv = ((res := qml.math.stack(res_list))[::2] - res[1::2]) / 2
        return qml.math.tensordot(psr_deriv, coeffs, axes=[[0], [0]])
    if single_measure and single_shot_entry:
        return _parshift_and_contract_single(results, qml.math.stack(coeffs))
    if single_measure or single_shot_entry:
        return tuple((_parshift_and_contract_single(r, qml.math.stack(coeffs)) for r in zip(*results)))
    return tuple((tuple((_parshift_and_contract_single(_r, qml.math.stack(coeffs)) for _r in zip(*r))) for r in zip(*results)))