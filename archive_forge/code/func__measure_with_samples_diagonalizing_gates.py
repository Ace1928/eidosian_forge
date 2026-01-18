from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def _measure_with_samples_diagonalizing_gates(mps: List[SampleMeasurement], state: np.ndarray, shots: Shots, is_state_batched: bool=False, rng=None, prng_key=None) -> TensorLike:
    """
    Returns the samples of the measurement process performed on the given state,
    by rotating the state into the measurement basis using the diagonalizing gates
    given by the measurement process.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    state = _apply_diagonalizing_gates(mps, state, is_state_batched)
    total_indices = len(state.shape) - is_state_batched
    wires = qml.wires.Wires(range(total_indices))

    def _process_single_shot(samples):
        processed = []
        for mp in mps:
            res = mp.process_samples(samples, wires)
            if not isinstance(mp, CountsMP):
                res = qml.math.squeeze(res)
            processed.append(res)
        return tuple(processed)
    if shots.has_partitioned_shots:
        processed_samples = []
        for s in shots:
            try:
                samples = sample_state(state, shots=s, is_state_batched=is_state_batched, wires=wires, rng=rng, prng_key=prng_key)
            except ValueError as e:
                if str(e) != 'probabilities contain NaN':
                    raise e
                samples = qml.math.full((s, len(wires)), 0)
            processed_samples.append(_process_single_shot(samples))
        return tuple(zip(*processed_samples))
    try:
        samples = sample_state(state, shots=shots.total_shots, is_state_batched=is_state_batched, wires=wires, rng=rng, prng_key=prng_key)
    except ValueError as e:
        if str(e) != 'probabilities contain NaN':
            raise e
        samples = qml.math.full((shots.total_shots, len(wires)), 0)
    return _process_single_shot(samples)