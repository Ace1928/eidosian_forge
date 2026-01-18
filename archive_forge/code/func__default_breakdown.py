import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _default_breakdown(qid: 'cirq.Qid') -> Tuple[Any, Any]:
    try:
        plane_pos = complex(qid)
        return (plane_pos.real, plane_pos.imag)
    except TypeError:
        return (None, qid)