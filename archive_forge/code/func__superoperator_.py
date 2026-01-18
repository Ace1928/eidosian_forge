import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _superoperator_(self) -> np.ndarray:
    """Returns superoperator representation of self if possible, else `NotImplemented`."""
    if not self._has_superoperator_():
        return NotImplemented
    return qis.kraus_to_superoperator(self._kraus_())