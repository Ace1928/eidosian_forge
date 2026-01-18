from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _convert_to_fsim(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.FSimGate]:
    theta = phi = None
    if isinstance(g, cirq.FSimGate) or (isinstance(g, cirq.PhasedFSimGate) and self._approx_eq_or_symbol(g._value_equality_values_()[1:4], (0.0, 0.0, 0.0))):
        theta = g.theta
        phi = g.phi
    if isinstance(g, cirq.PhasedISwapPowGate) and self._approx_eq_or_symbol(g.phase_exponent, 0):
        g = g._iswap
    if isinstance(g, (cirq.ISwapPowGate, cirq.CZPowGate)):
        if not self._approx_eq_or_symbol(_exp(np.pi * 1j * g.global_shift * g.exponent), 1.0):
            return None
        theta = -g.exponent * np.pi / 2 if isinstance(g, cirq.ISwapPowGate) else 0
        phi = -g.exponent * np.pi if isinstance(g, cirq.CZPowGate) else 0
    if isinstance(g, cirq.IdentityGate):
        theta = phi = 0
    return None if theta is None or phi is None else cirq.FSimGate(theta, phi)