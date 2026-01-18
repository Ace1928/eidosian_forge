from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
def _gate_str(self, gettr: Callable[[Any], str]=str) -> str:
    return _gate_str(self.gate, gettr)