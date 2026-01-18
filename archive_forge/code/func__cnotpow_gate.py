import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _cnotpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.CNotPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('CNOT {0} {1}\n', op.qubits[0], op.qubits[1])
    return None