import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@cached_property
def _control_keys(self) -> FrozenSet['cirq.MeasurementKey']:
    keys = frozenset() if not protocols.control_keys(self.circuit) else protocols.control_keys(self._mapped_single_loop())
    if self.repeat_until is not None:
        keys |= frozenset(self.repeat_until.keys) - self._measurement_key_objs_()
    return keys