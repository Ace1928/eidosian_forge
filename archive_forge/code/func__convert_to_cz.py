from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _convert_to_cz(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.CZPowGate]:
    if isinstance(g, cirq.CZPowGate):
        return g
    cg = self._convert_to_fsim(g)
    return None if cg is None or not self._approx_eq_or_symbol(cg.theta, 0) else cirq.CZPowGate(exponent=-cg.phi / np.pi)