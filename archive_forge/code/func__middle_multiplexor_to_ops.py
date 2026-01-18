from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
def _middle_multiplexor_to_ops(q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, eigvals: np.ndarray):
    theta = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    angles = _multiplexed_angles(theta)
    rzs = [cirq.rz(angle).on(q0) for angle in angles]
    ops = [rzs[0], cirq.CNOT(q1, q0), rzs[1], cirq.CNOT(q2, q0), rzs[2], cirq.CNOT(q1, q0), rzs[3], cirq.CNOT(q2, q0)]
    return _optimize_multiplexed_angles_circuit(ops)