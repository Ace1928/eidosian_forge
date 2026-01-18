import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
class DAGOutNode(DAGNode):
    """Object to represent an outgoing wire node in the DAGCircuit."""
    __slots__ = ['wire', 'sort_key']

    def __init__(self, wire):
        """Create an outgoing node"""
        super().__init__()
        self.wire = wire
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGOutNode"""
        return f'DAGOutNode(wire={self.wire})'