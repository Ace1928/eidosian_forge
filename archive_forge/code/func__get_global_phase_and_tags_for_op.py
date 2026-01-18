import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _get_global_phase_and_tags_for_op(op: 'cirq.Operation') -> Tuple[Optional[complex], List[Any]]:
    if isinstance(op.gate, ops.GlobalPhaseGate):
        return (complex(op.gate.coefficient), list(op.tags))
    elif isinstance(op.untagged, CircuitOperation):
        op_phase, op_tags = _get_global_phase_and_tags_for_ops(op.untagged.circuit.all_operations())
        return (op_phase, list(op.tags) + op_tags)
    return (None, [])