import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _sorted_operations_(self) -> Tuple['cirq.Operation', ...]:
    if self._sorted_operations is None:
        self._sorted_operations = tuple(sorted(self._operations, key=lambda op: op.qubits))
    return self._sorted_operations