from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def _measure_classical_shadow(mp: List[Union[ClassicalShadowMP, ShadowExpvalMP]], state: np.ndarray, shots: Shots, is_state_batched: bool=False, rng=None, prng_key=None):
    """
    Returns the result of a classical shadow measurement on the given state.

    A classical shadow measurement doesn't fit neatly into the current measurement API
    since different diagonalizing gates are used for each shot. Here it's treated as a
    state measurement with shots instead of a sample measurement.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    mp = mp[0]
    wires = qml.wires.Wires(range(len(state.shape)))
    if shots.has_partitioned_shots:
        return [tuple((mp.process_state_with_shots(state, wires, s, rng=rng) for s in shots))]
    return [mp.process_state_with_shots(state, wires, shots.total_shots, rng=rng)]