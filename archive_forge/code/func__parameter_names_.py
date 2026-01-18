from typing import AbstractSet, Any, Optional, Tuple
import numpy as np
import cirq
from cirq._compat import proper_repr
def _parameter_names_(self) -> AbstractSet[str]:
    return cirq.parameter_names(self.hold_time) | cirq.parameter_names(self.coupling_mhz) | cirq.parameter_names(self.rise_time) | cirq.parameter_names(self.padding_time) | cirq.parameter_names(self.q0_detune_mhz) | cirq.parameter_names(self.q1_detune_mhz)