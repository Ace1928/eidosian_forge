from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
@dataclasses.dataclass
class MeasureInfo:
    """Extra info about a single measurement within a circuit.

    Attributes:
        key: String identifying this measurement.
        qubits: List of measured qubits, in order.
        instances: The number of times a given key occurs in a circuit.
        invert_mask: a list of booleans describing whether the results should
            be flipped for each of the qubits in the qubits field.
        tags: Tags applied to this measurement gate.
    """
    key: str
    qubits: List[cirq.GridQubit]
    instances: int
    invert_mask: List[bool]
    tags: List[Hashable]