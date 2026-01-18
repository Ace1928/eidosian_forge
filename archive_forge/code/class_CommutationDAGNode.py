import heapq
from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane.transforms import transform
class CommutationDAGNode:
    """Class to store information about a quantum operation in a node of the
    commutation DAG.

    Args:
        op (.Operation): PennyLane operation.
        wires (.Wires): Wires on which the operation acts on.
        node_id (int): ID of the node in the DAG.
        successors (array[int]): List of the node's successors in the DAG.
        predecessors (array[int]): List of the node's predecessors in the DAG.
        reachable (bool): Attribute used to check reachability by pairwise commutation.
    """
    __slots__ = ['op', 'wires', 'target_wires', 'control_wires', 'node_id', 'successors', 'predecessors', 'reachable']

    def __init__(self, op=None, wires=None, target_wires=None, control_wires=None, successors=None, predecessors=None, reachable=None, node_id=-1):
        self.op = op
        'Operation: The operation represented by the nodes.'
        self.wires = wires
        'Wires: The wires that the operation acts on.'
        self.target_wires = target_wires
        'Wires: The target wires of the operation.'
        self.control_wires = control_wires if control_wires is not None else []
        'Wires: The control wires of the operation.'
        self.node_id = node_id
        'int: The ID of the operation in the DAG.'
        self.successors = successors if successors is not None else []
        "list(int): List of the node's successors."
        self.predecessors = predecessors if predecessors is not None else []
        "list(int): List of the node's predecessors."
        self.reachable = reachable
        'bool: Useful attribute to create the commutation DAG.'