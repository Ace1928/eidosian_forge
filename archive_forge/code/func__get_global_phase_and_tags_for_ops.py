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
def _get_global_phase_and_tags_for_ops(op_list: Any) -> Tuple[Optional[complex], List[Any]]:
    global_phase: Optional[complex] = None
    tags: List[Any] = []
    for op in op_list:
        op_phase, op_tags = _get_global_phase_and_tags_for_op(op)
        if op_phase:
            if global_phase is None:
                global_phase = complex(1)
            global_phase *= op_phase
        if op_tags:
            tags.extend(op_tags)
    return (global_phase, tags)