from typing import (
import numpy as np
from cirq import protocols, _compat
from cirq.circuits import AbstractCircuit, Alignment, Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType
@_compat.cached_property
def _all_operations(self) -> Tuple['cirq.Operation', ...]:
    return tuple(super().all_operations())