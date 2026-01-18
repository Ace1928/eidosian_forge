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
def _expval_pulse_odegen(tape, argnum, atol):
    """Compute the pulse generator parameter-shift rule for a quantum circuit that returns expectation
    values of observables.

    Args:
        tape (`~.QuantumTape`): Quantum circuit to be differentiated with the pulse generator
            parameter-shift rule.
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        atol (float): absolute tolerance used to determine vanishing contributions.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian.

    """
    argnum = argnum or tape.trainable_params
    cache = {'total_num_tapes': 0}
    gradient_data = []
    gradient_tapes = []
    tape_params = tape.get_parameters()
    for idx, param in enumerate(tape_params):
        shape = qml.math.shape(param)
        if idx not in argnum:
            gradient_data.append((0, 0, None, shape))
            continue
        tapes, data, cache = _generate_tapes_and_coeffs(tape, idx, atol, cache)
        gradient_data.append((*data, shape))
        gradient_tapes.extend(tapes)
    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    partitioned_shots = tape.shots.has_partitioned_shots
    tape_specs = (single_measure, num_params, num_measurements, tape.shots)

    def processing_fn(results):
        """Post-process the results of the parameter-shifted tapes for
        ``pulse_odegen`` into the gradient."""
        grads = []
        zero_parshapes = []
        for start, end, coeffs, par_shape in gradient_data:
            if start == end:
                grads.append(None)
                zero_parshapes.append(par_shape)
                continue
            res = results[start:end]
            g = _parshift_and_contract(res, coeffs, single_measure, not partitioned_shots)
            grads.append(g)
            nonzero_parshape = par_shape
        zero_parshapes = iter(zero_parshapes)
        for i, _g in enumerate(grads):
            if _g is None:
                par_shapes = (nonzero_parshape, next(zero_parshapes))
                zero_rep = _make_zero_rep(g, single_measure, partitioned_shots, par_shapes)
                grads[i] = zero_rep
        return reorder_grads(grads, tape_specs)
    return (gradient_tapes, processing_fn)