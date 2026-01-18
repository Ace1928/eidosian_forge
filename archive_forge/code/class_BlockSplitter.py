from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
class BlockSplitter:
    """Splits a block of nodes into sub-blocks over disjoint qubits.
    The implementation is based on the Disjoint Set Union data structure."""

    def __init__(self):
        self.leader = {}
        self.group = {}

    def find_leader(self, index):
        """Find in DSU."""
        if index not in self.leader:
            self.leader[index] = index
            self.group[index] = []
            return index
        if self.leader[index] == index:
            return index
        self.leader[index] = self.find_leader(self.leader[index])
        return self.leader[index]

    def union_leaders(self, index1, index2):
        """Union in DSU."""
        leader1 = self.find_leader(index1)
        leader2 = self.find_leader(index2)
        if leader1 == leader2:
            return
        if len(self.group[leader1]) < len(self.group[leader2]):
            leader1, leader2 = (leader2, leader1)
        self.leader[leader2] = leader1
        self.group[leader1].extend(self.group[leader2])
        self.group[leader2].clear()

    def run(self, block):
        """Splits block of nodes into sub-blocks over disjoint qubits."""
        for node in block:
            indices = node.qargs
            if not indices:
                continue
            first = indices[0]
            for index in indices[1:]:
                self.union_leaders(first, index)
            self.group[self.find_leader(first)].append(node)
        blocks = []
        for index in self.leader:
            if self.leader[index] == index:
                blocks.append(self.group[index])
        return blocks