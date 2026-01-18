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
def _group_until_different(items: Iterable[_TIn], key: Callable[[_TIn], _TKey], val=lambda e: e):
    """Groups runs of items that are identical according to a keying function.

    Args:
        items: The items to group.
        key: If two adjacent items produce the same output from this function,
            they will be grouped.
        val: Maps each item into a value to put in the group. Defaults to the
            item itself.

    Examples:
        _group_until_different(range(11), key=is_prime) yields
            (False, [0, 1])
            (True, [2, 3])
            (False, [4])
            (True, [5])
            (False, [6])
            (True, [7])
            (False, [8, 9, 10])

    Yields:
        Tuples containing the group key and item values.
    """
    return ((k, [val(i) for i in v]) for k, v in itertools.groupby(items, key))