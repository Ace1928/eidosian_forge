from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def _gate_to_quirk_op(gate: ops.Gate) -> Optional[QuirkOp]:
    for gate_type, func in _known_gate_conversions.items():
        if isinstance(gate, gate_type):
            return func(gate)
    return None