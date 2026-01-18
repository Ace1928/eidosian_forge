import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _xxpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.XPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('X {0}\nX {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format('RX({0}) {1}\nRX({2}) {3}\n', gate._exponent * np.pi, op.qubits[0], gate._exponent * np.pi, op.qubits[1])