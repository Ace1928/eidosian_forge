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
def _tapes_data_hardware(tape, operation, key, num_split_times, use_broadcasting):
    """Create tapes and gradient data for a trainable parameter of a HardwareHamiltonian,
    taking into account its reordering function.

    Args:
        tape (QuantumScript): Tape for which to compute the stochastic pulse parameter-shift
            gradient tapes.
        operation (tuple[Operation, int, int]): Information about the pulse operation to be
            shifted. The first entry is the operation itself, the second entry is its position
            in the ``tape``, and the third entry is the index of the differentiated parameter
            within the ``HardwareHamiltonian`` of the operation.
        key (tuple[int]): Randomness key to create spliting times in ``_generate_tapes_and_cjacs``
        num_split_times (int): Number of splitting times at which to create shifted tapes for
            the stochastic shift rule.
        use_broadcasting (bool): Whether to use broadcasting in the shift rule or not.

    Returns:
        list[QuantumScript]: Gradient tapes for the indicated operation and Hamiltonian term.
        tuple: Gradient postprocessing data.
            See comment below.

    This function analyses the ``reorder_fn`` of the ``HardwareHamiltonian`` of the pulse
    that is being differentiated. Given a ``term_idx``, the index of the parameter
    in the Hamiltonian, stochastic parameter shift tapes are created for all terms in the
    Hamiltonian into which the parameter feeds. While this is a one-to-one relation for
    standard ``ParametrizedHamiltonian`` objects, the reordering function of
    the ``HardwareHamiltonian`` requires to create tapes for multiple Hamiltonian terms,
    and for each term ``_generate_tapes_and_cjacs`` is called.

    The returned gradient data has four entries:

      1. ``int``: Total number of tapes created for all the terms that depend on the indicated
         parameter.
      2. ``tuple[tensor_like]``: Classical Jacobians for all terms and splitting times
      3. ``float``: Prefactor for the Monte Carlo estimate of the integral in the stochastic
         shift rule.
      4. ``tuple[tensor_like]``: Parameter-shift coefficients for all terms.

    The tuple axes in the second and fourth entry correspond to the different terms in the
    Hamiltonian.
    """
    op, op_idx, term_idx = operation
    fake_params, allowed_outputs = (np.arange(op.num_params), set(range(op.num_params)))
    reordered = op.H.reorder_fn(fake_params, op.H.coeffs_parametrized)

    def _raise():
        raise ValueError(f'Only permutations, fan-out or fan-in functions are allowed as reordering functions in HardwareHamiltonians treated by stoch_pulse_grad. The reordering function of {op.H} mapped {fake_params} to {reordered}.')
    cjacs, tapes, psr_coeffs = ([], [], [])
    for coeff_idx, x in enumerate(reordered):
        if not hasattr(x, '__len__'):
            if x not in allowed_outputs:
                _raise()
            if x != term_idx:
                continue
            cjac_idx = None
        else:
            if not all((_x in list(range(op.num_params)) for _x in x)):
                _raise()
            if term_idx not in x:
                continue
            cjac_idx = np.argwhere([_x == term_idx for _x in x])[0][0]
        _operation = (op, op_idx, coeff_idx)
        _tapes, _cjacs, int_prefactor, _psr_coeffs = _generate_tapes_and_cjacs(tape, _operation, key, num_split_times, use_broadcasting, cjac_idx)
        cjacs.append(qml.math.stack(_cjacs))
        tapes.extend(_tapes)
        psr_coeffs.append(_psr_coeffs)
    data = (len(tapes), tuple(cjacs), int_prefactor, tuple(psr_coeffs))
    return (tapes, data)