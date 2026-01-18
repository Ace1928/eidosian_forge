from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
@lru_cache()
def _qpos_y(qubits: Union[cirq.Qid, Tuple[cirq.Qid, ...]], all_qubits: Tuple[cirq.Qid, ...]) -> float:
    """Given a qubit or qubits, return the position y value (used for drawing).

    For multiple qubits, the position is the mean of the qubit indices.
    This "flips" the coordinate so qubit 0 is at the maximal y position.

    Args:
        qubits: The qubits involved in the tensor.
        all_qubits: All qubits in the circuit, allowing us
            to position the zero'th qubit at the top.
    """
    if isinstance(qubits, cirq.Qid):
        return _qpos_y((qubits,), all_qubits)
    pos = [all_qubits.index(q) for q in qubits]
    x = np.mean(pos).item()
    return len(all_qubits) - x - 1