from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _convert_to_identity(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.IdentityGate]:
    cg = self._convert_to_fsim(g)
    return None if cg is None or not self._approx_eq_or_symbol((cg.theta, cg.phi), (0, 0)) else cirq.IdentityGate(2)