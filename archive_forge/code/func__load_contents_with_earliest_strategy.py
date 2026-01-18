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
def _load_contents_with_earliest_strategy(self, contents: 'cirq.OP_TREE'):
    """Optimized algorithm to load contents quickly.

        The default algorithm appends operations one-at-a-time, letting them
        fall back until they encounter a moment they cannot commute with. This
        is slow because it requires re-checking for conflicts at each moment.

        Here, we instead keep track of the greatest moment that contains each
        qubit, measurement key, and control key, and append the operation to
        the moment after the maximum of these. This avoids having to check each
        moment.

        Args:
            contents: The initial list of moments and operations defining the
                circuit. You can also pass in operations, lists of operations,
                or generally anything meeting the `cirq.OP_TREE` contract.
                Non-moment entries will be inserted according to the EARLIEST
                insertion strategy.
        """
    qubit_indices: Dict['cirq.Qid', int] = {}
    mkey_indices: Dict['cirq.MeasurementKey', int] = {}
    ckey_indices: Dict['cirq.MeasurementKey', int] = {}
    op_lists_by_index: Dict[int, List['cirq.Operation']] = defaultdict(list)
    moments_by_index: Dict[int, 'cirq.Moment'] = {}
    length = 0
    for mop in ops.flatten_to_ops_or_moments(contents):
        placement_index = get_earliest_accommodating_moment_index(mop, qubit_indices, mkey_indices, ckey_indices, length)
        length = max(length, placement_index + 1)
        if isinstance(mop, Moment):
            moments_by_index[placement_index] = mop
        else:
            op_lists_by_index[placement_index].append(mop)
    for i in range(length):
        if i in moments_by_index:
            self._moments.append(moments_by_index[i].with_operations(op_lists_by_index[i]))
        else:
            self._moments.append(Moment(op_lists_by_index[i]))