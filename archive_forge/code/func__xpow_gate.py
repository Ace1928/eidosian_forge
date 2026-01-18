import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _xpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.XPowGate, op.gate)
    if gate._exponent == 1 and gate._global_shift != -0.5:
        return formatter.format('X {0}\n', op.qubits[0])
    return formatter.format('RX({0}) {1}\n', gate._exponent * np.pi, op.qubits[0])