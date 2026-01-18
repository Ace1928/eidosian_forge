import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def _node_and_coordinates(nodes: Iterable[_GRIDLIKE_NODE]) -> Iterable[Tuple[_GRIDLIKE_NODE, Tuple[int, int]]]:
    """Yield tuples whose first element is the input node and the second is guaranteed to be a tuple
    of two integers. The input node can be a tuple of ints or a GridQubit."""
    for node in nodes:
        if isinstance(node, GridQubit):
            yield (node, (node.row, node.col))
        else:
            x, y = node
            yield (node, (x, y))