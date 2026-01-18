import itertools
import dataclasses
import inspect
from collections import defaultdict
from typing import (
from typing_extensions import runtime_checkable
from typing_extensions import Protocol
from cirq import devices, ops
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
@dataclasses.dataclass(frozen=True)
class DecompositionContext:
    """Stores common configurable options for decomposing composite gates into simpler operations.

    Args:
        qubit_manager: A `cirq.QubitManager` instance to allocate clean / dirty ancilla qubits as
            part of the decompose protocol.
    """
    qubit_manager: 'cirq.QubitManager'