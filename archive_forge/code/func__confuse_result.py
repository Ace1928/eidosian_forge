import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
def _confuse_result(self, bits: List[int], qubits: Sequence['cirq.Qid'], confusion_map: Dict[Tuple[int, ...], np.ndarray]):
    """Applies confusion matrices to measured results.

        Compare with _confuse_results in cirq-core/cirq/sim/simulator.py.
        """
    confused = list(bits)
    dims = [q.dimension for q in qubits]
    for indices, confuser in confusion_map.items():
        mat_dims = [dims[k] for k in indices]
        row = value.big_endian_digits_to_int((bits[k] for k in indices), base=mat_dims)
        new_val = self.prng.choice(len(confuser), p=confuser[row])
        new_bits = value.big_endian_int_to_digits(new_val, base=mat_dims)
        for i, k in enumerate(indices):
            confused[k] = new_bits[i]
    return confused