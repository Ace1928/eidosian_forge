import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _hpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.HPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('H {0}\n', op.qubits[0])
    return formatter.format('RY({0}) {3}\nRX({1}) {3}\nRY({2}) {3}\n', 0.25 * np.pi, gate._exponent * np.pi, -0.25 * np.pi, op.qubits[0])