from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _convert_to_phased_fsim(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.PhasedFSimGate]:
    if isinstance(g, cirq.PhasedFSimGate):
        return g
    chi = 0.0
    if isinstance(g, cirq.PhasedISwapPowGate):
        chi = g.phase_exponent * 2 * np.pi
        g = g._iswap
    fsim = self._convert_to_fsim(g)
    return None if fsim is None else cirq.PhasedFSimGate(fsim.theta, 0, chi, 0, fsim.phi)