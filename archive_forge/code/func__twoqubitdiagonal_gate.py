import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _twoqubitdiagonal_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.TwoQubitDiagonalGate, op.gate)
    diag_angles_radians = np.asarray(gate._diag_angles_radians)
    if np.count_nonzero(diag_angles_radians) == 1:
        if diag_angles_radians[0] != 0:
            return formatter.format('CPHASE00({0}) {1} {2}\n', diag_angles_radians[0], op.qubits[0], op.qubits[1])
        elif diag_angles_radians[1] != 0:
            return formatter.format('CPHASE01({0}) {1} {2}\n', diag_angles_radians[1], op.qubits[0], op.qubits[1])
        elif diag_angles_radians[2] != 0:
            return formatter.format('CPHASE10({0}) {1} {2}\n', diag_angles_radians[2], op.qubits[0], op.qubits[1])
        elif diag_angles_radians[3] != 0:
            return formatter.format('CPHASE({0}) {1} {2}\n', diag_angles_radians[3], op.qubits[0], op.qubits[1])
    return None