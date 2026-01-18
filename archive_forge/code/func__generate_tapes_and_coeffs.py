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
def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    """Compute the modified tapes and coefficients required to compute the pulse generator
    derivative of a tape with respect to an indicated trainable parameter.

    Args:
        tape (`~.QuantumTape`): Tape to differentiate.
        idx (int): Index (referring to ``tape.trainable_parameters``) of the parameter
            with respect to which to differentiate.
        atol (float): absolute tolerance used to determine whether a coefficient is zero.
        cache (dict): Caching dictionary that allows to skip adding duplicate modified tapes.

    Returns:
        list[`~.QuantumScript`]: Modified tapes to be added to the pulse generator differentiation
            tapes. It is an empty list if modified tapes were already created for another
            parameter of the pulse of interest.
        tuple[int, int, tensor_like]: Gradient computation data, consisting of the start and end
            indices into the total list of tapes as well as the coefficients that need to be
            contracted with the corresponding results to obtain the partial derivative with respect
            to the indicated trainable parameter.
        dict: Input ``cache`` dictionary. If the cache lookup missed, the cache is extended by one
            entry and its entry ``"total_num_tapes"`` is increased by the number of created tapes.
    """
    op, op_idx, term_idx = tape.get_operation(idx)
    if op_idx in cache:
        start, end, all_coeffs = cache[op_idx]
        return ([], (start, end, all_coeffs[term_idx]), cache)
    if not isinstance(op, ParametrizedEvolution):
        raise ValueError(f'pulse_odegen does not support differentiating parameters of other operations than pulses, but received operation {op}.')
    num_wires = len(op.wires)
    generators = _one_parameter_generators(op)
    all_coeffs = _one_parameter_paulirot_coeffs(generators, num_wires)
    all_coeffs, pauli_words = _nonzero_coeffs_and_words(all_coeffs, num_wires, atol)
    pauli_rots = [qml.PauliRot(angle, word, wires=op.wires) for word in pauli_words for angle in [np.pi / 2, -np.pi / 2]]
    tapes = _insert_op(tape, pauli_rots, op_idx)
    end = (start := cache['total_num_tapes']) + (num_tapes := len(tapes))
    cache[op_idx] = (start, end, all_coeffs)
    cache['total_num_tapes'] += num_tapes
    return (tapes, (start, end, all_coeffs[term_idx]), cache)