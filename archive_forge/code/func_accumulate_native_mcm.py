from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def accumulate_native_mcm(circuit: qml.tape.QuantumScript, all_shot_meas, one_shot_meas):
    """Incorporates new measurements in current measurement sequence.

    Args:
        circuit (QuantumTape): A one-shot (auxiliary) ``QuantumScript``
        all_shot_meas (Sequence[Any]): List of accumulated measurement results
        one_shot_meas (Sequence[Any]): List of measurement results

    Returns:
        tuple(TensorLike): The results of the simulation
    """
    if len(circuit.measurements) == 1:
        one_shot_meas = [one_shot_meas]
    if all_shot_meas is None:
        new_shot_meas = list(one_shot_meas)
        for i, (m, s) in enumerate(zip(circuit.measurements, new_shot_meas)):
            if isinstance(m, SampleMP) and isinstance(s, np.ndarray):
                new_shot_meas[i] = [s]
        return new_shot_meas
    new_shot_meas = all_shot_meas
    for i, m in enumerate(circuit.measurements):
        if isinstance(m, CountsMP):
            tmp = Counter(all_shot_meas[i])
            tmp.update(Counter(one_shot_meas[i]))
            new_shot_meas[i] = tmp
        elif isinstance(m, (ExpectationMP, ProbabilityMP)):
            new_shot_meas[i] = all_shot_meas[i] + one_shot_meas[i]
        elif isinstance(m, SampleMP):
            new_shot_meas[i].append(one_shot_meas[i])
        else:
            raise TypeError(f'Unsupported measurement of {type(m).__name__}.')
    return new_shot_meas