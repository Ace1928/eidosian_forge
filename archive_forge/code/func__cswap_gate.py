import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _cswap_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    return formatter.format('CSWAP {0} {1} {2}\n', op.qubits[0], op.qubits[1], op.qubits[2])