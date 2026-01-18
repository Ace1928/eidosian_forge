import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _yypow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.YYPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('Y {0}\nY {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format('RY({0}) {1}\nRY({2}) {3}\n', gate._exponent * np.pi, op.qubits[0], gate._exponent * np.pi, op.qubits[1])