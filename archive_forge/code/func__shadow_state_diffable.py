import warnings
from functools import reduce, partial
from itertools import product
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ClassicalShadowMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
def _shadow_state_diffable(tape, wires):
    """Differentiable version of the shadow state transform"""
    wires_list = wires if isinstance(wires[0], list) else [wires]
    if any((len(w) >= 8 for w in wires_list)):
        warnings.warn('Differentiable state reconstruction for more than 8 qubits is not recommended', UserWarning)
    all_observables = []
    for w in wires_list:
        observables = []
        for obs in product(*[[qml.Identity, qml.X, qml.Y, qml.Z] for _ in range(len(w))]):
            observables.append(reduce(lambda a, b: a @ b, [ob(wire) for ob, wire in zip(obs, w)]))
        all_observables.extend(observables)
    tapes, _ = _replace_obs(tape, qml.shadow_expval, all_observables)

    def post_processing_fn(results):
        """Post process the classical shadows."""
        results = results[0]
        results = qml.math.cast(results, np.complex64)
        states = []
        start = 0
        for w in wires_list:
            obs_matrices = qml.math.stack([qml.math.cast_like(qml.math.convert_like(qml.matrix(obs), results), results) for obs in all_observables[start:start + 4 ** len(w)]])
            s = qml.math.einsum('a,abc->bc', results[start:start + 4 ** len(w)], obs_matrices) / 2 ** len(w)
            states.append(s)
            start += 4 ** len(w)
        return states if isinstance(wires[0], list) else states[0]
    return (tapes, post_processing_fn)