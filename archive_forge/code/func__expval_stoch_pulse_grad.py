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
def _expval_stoch_pulse_grad(tape, argnum, num_split_times, key, use_broadcasting):
    """Compute the gradient of a quantum circuit composed of pulse sequences that measures
    an expectation value or probabilities, by applying the stochastic parameter shift rule.
    See the main function for the signature.
    """
    tapes = []
    gradient_data = []
    for idx, trainable_idx in enumerate(tape.trainable_params):
        if trainable_idx not in argnum:
            gradient_data.append((0, None, None, None))
            continue
        key, _key = jax.random.split(key)
        operation = tape.get_operation(idx)
        op, *_ = operation
        if not isinstance(op, ParametrizedEvolution):
            raise ValueError('stoch_pulse_grad does not support differentiating parameters of other operations than pulses.')
        if isinstance(op.H, HardwareHamiltonian):
            _tapes, data = _tapes_data_hardware(tape, operation, key, num_split_times, use_broadcasting)
        else:
            _tapes, cjacs, int_prefactor, psr_coeffs = _generate_tapes_and_cjacs(tape, operation, _key, num_split_times, use_broadcasting)
            data = (len(_tapes), qml.math.stack(cjacs), int_prefactor, psr_coeffs)
        tapes.extend(_tapes)
        gradient_data.append(data)
    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    has_partitioned_shots = tape.shots.has_partitioned_shots
    tape_specs = (single_measure, num_params, num_measurements, tape.shots)

    def processing_fn(results):
        start = 0
        grads = []
        for num_tapes, cjacs, int_prefactor, psr_coeffs in gradient_data:
            if num_tapes == 0:
                grads.append(None)
                continue
            res = results[start:start + num_tapes]
            start += num_tapes
            g = _parshift_and_integrate(res, cjacs, int_prefactor, psr_coeffs, single_measure, has_partitioned_shots, use_broadcasting)
            grads.append(g)
        zero_rep = _make_zero_rep(g, single_measure, has_partitioned_shots)
        grads = [zero_rep if g is None else g for g in grads]
        return reorder_grads(grads, tape_specs)
    return (tapes, processing_fn)