from typing import List, Callable, TYPE_CHECKING
from scipy.linalg import cossin
import numpy as np
from cirq import ops
from cirq.linalg import decompositions, predicates
def _multiplexed_cossin(cossin_qubits: 'List[cirq.Qid]', angles: List[float], rot_func: Callable=ops.ry) -> 'op_tree.OpTree':
    """Performs a multiplexed rotation over all qubits in this unitary matrix,

    Uses ry and rz multiplexing for quantum shannon decomposition

    Args:
        cossin_qubits: Subset of total qubits involved in this unitary gate
        angles: List of angles to be multiplexed over for the given type of rotation
        rot_func: Rotation function used for this multiplexing implementation
                    (cirq.ry or cirq.rz)

    Calls:
        No major calls

    Yields: Single operation from OP TREE from set 1- and 2-qubit gates: {ry,rz,CNOT}
    """
    main_qubit = cossin_qubits[0]
    control_qubits = cossin_qubits[1:]
    for j in range(len(angles)):
        rotation = sum((-angle if bin(_nth_gray(j) & i).count('1') % 2 else angle for i, angle in enumerate(angles)))
        rotation = rotation * 2 / len(angles)
        select_string = _nth_gray(j) ^ _nth_gray(j + 1)
        select_qubit = next((i for i in range(len(angles)) if select_string >> i & 1))
        select_qubit = max(-select_qubit - 1, -len(control_qubits))
        yield rot_func(rotation).on(main_qubit)
        yield ops.CNOT(control_qubits[select_qubit], main_qubit)