import re
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types, gate_features, control_values as cv
from cirq.type_workarounds import NotImplementedType
def _group_interchangeable_qubits(self) -> Tuple[Union['cirq.Qid', Tuple[int, FrozenSet['cirq.Qid']]], ...]:
    if not isinstance(self.gate, gate_features.InterchangeableQubitsGate):
        return self.qubits
    groups: Dict[int, List['cirq.Qid']] = {}
    for i, q in enumerate(self.qubits):
        k = self.gate.qubit_index_to_equivalence_group_key(i)
        groups.setdefault(k, []).append(q)
    return tuple(sorted(((k, frozenset(v)) for k, v in groups.items())))