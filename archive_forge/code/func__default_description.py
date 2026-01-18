from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _default_description(self) -> str:
    return f'`cirq_google.FSimGateFamily` which accepts any instance of gate types in\ngate_types_to_check: {_gates_to_str(self.gate_types_to_check)}\nwhich matches (across types), via instance check / value equality, a gate in\ngates_to_accept: {_gates_to_str(self.gates_to_accept)}'