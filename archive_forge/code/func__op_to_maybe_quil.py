import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _op_to_maybe_quil(self, op: cirq.Operation) -> Optional[str]:
    for gate_type in SUPPORTED_GATES.keys():
        if isinstance(op.gate, gate_type):
            quil: Callable[[cirq.Operation, QuilFormatter], Optional[str]] = SUPPORTED_GATES[gate_type]
            return quil(op, self.formatter)
    return None