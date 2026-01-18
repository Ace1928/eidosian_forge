import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _cczpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.CCZPowGate, op.gate)
    if gate._exponent != 1:
        return None
    lines = [formatter.format('H {0}\n', op.qubits[2]), formatter.format('CCNOT {0} {1} {2}\n', op.qubits[0], op.qubits[1], op.qubits[2]), formatter.format('H {0}\n', op.qubits[2])]
    return ''.join(lines)