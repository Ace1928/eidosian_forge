import re
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types, gate_features, control_values as cv
from cirq.type_workarounds import NotImplementedType
def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
    new_gate = protocols.with_key_path_prefix(self.gate, prefix)
    if new_gate is NotImplemented:
        return NotImplemented
    if new_gate is self.gate:
        return self
    return new_gate.on(*self.qubits)