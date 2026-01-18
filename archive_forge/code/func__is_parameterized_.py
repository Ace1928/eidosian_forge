from typing import AbstractSet, Any, Optional, Tuple
import numpy as np
import cirq
from cirq._compat import proper_repr
def _is_parameterized_(self) -> bool:
    return cirq.is_parameterized(self.hold_time) or cirq.is_parameterized(self.coupling_mhz) or cirq.is_parameterized(self.rise_time) or cirq.is_parameterized(self.padding_time) or cirq.is_parameterized(self.q0_detune_mhz) or cirq.is_parameterized(self.q1_detune_mhz)