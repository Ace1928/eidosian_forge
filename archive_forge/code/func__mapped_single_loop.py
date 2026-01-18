import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def _mapped_single_loop(self, repetition_id: Optional[str]=None) -> 'cirq.Circuit':
    circuit = self._mapped_any_loop
    if repetition_id:
        circuit = protocols.with_rescoped_keys(circuit, (repetition_id,))
    return protocols.with_rescoped_keys(circuit, self.parent_path, bindable_keys=self._extern_keys)