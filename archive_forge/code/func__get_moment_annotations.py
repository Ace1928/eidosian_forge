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
def _get_moment_annotations(moment: 'cirq.Moment') -> Iterator['cirq.Operation']:
    for op in moment.operations:
        if op.qubits:
            continue
        op = op.untagged
        if isinstance(op.gate, ops.GlobalPhaseGate):
            continue
        if isinstance(op, CircuitOperation):
            for m in op.circuit:
                yield from _get_moment_annotations(m)
        else:
            yield op