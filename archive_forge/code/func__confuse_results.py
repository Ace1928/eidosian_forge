import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def _confuse_results(self, bits: np.ndarray, qubits: Sequence['cirq.Qid'], confusion_map: Dict[Tuple[int, ...], np.ndarray], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> None:
    """Mutates `bits` using the confusion_map.

        Compare with _confuse_result in cirq-core/cirq/sim/simulation_state.py.
        """
    prng = value.parse_random_state(seed)
    for rep in bits:
        dims = [q.dimension for q in qubits]
        for indices, confuser in confusion_map.items():
            mat_dims = [dims[k] for k in indices]
            row = value.big_endian_digits_to_int((rep[k] for k in indices), base=mat_dims)
            new_val = prng.choice(len(confuser), p=confuser[row])
            new_bits = value.big_endian_int_to_digits(new_val, base=mat_dims)
            for i, k in enumerate(indices):
                rep[k] = new_bits[i]