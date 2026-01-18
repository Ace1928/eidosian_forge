from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _remove_successive_identity(self, dag, qubit, from_idx=None):
    """remove gates that have the same set of target qubits, follow each
            other immediately on these target qubits, and combine to the
            identity (consider sequences of length 2 for now)
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
            qubit (Qubit): qubit cache to inspect
            from_idx (int): only gates whose indexes in the cache are larger
                            than this value can be removed
        """
    i = 0
    while i < len(self.gatecache[qubit]) - 1:
        append = True
        node1 = self.gatecache[qubit][i]
        node2 = self.gatecache[qubit][i + 1]
        trgtqb1 = self._seperate_ctrl_trgt(node1)[2]
        trgtqb2 = self._seperate_ctrl_trgt(node2)[2]
        i += 1
        if trgtqb1 != trgtqb2:
            continue
        try:
            for qbt in trgtqb1:
                idx = self.gatecache[qbt].index(node1)
                if self.gatecache[qbt][idx + 1] is not node2:
                    append = False
        except (IndexError, ValueError):
            continue
        seq = [node1, node2]
        if append and self._is_identity(seq) and self._seq_as_one(seq):
            i += 1
            for node in seq:
                dag.remove_op_node(node)
                if from_idx is None or self.gatecache[qubit].index(node) > from_idx:
                    for qbt in node.qargs:
                        self.gatecache[qbt].remove(node)