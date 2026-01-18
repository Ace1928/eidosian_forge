from typing import Tuple
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def _decompose_to_2q(self, dag: DAGCircuit, op: PauliEvolutionGate) -> DAGCircuit:
    """Decompose the SparsePauliOp into two-qubit.

        Args:
            dag: The dag needed to get access to qubits.
            op: The operator with all the Pauli terms we need to apply.

        Returns:
            A dag made of two-qubit :class:`.PauliEvolutionGate`.
        """
    sub_dag = dag.copy_empty_like()
    required_paulis = {self._pauli_to_edge(pauli): (pauli, coeff) for pauli, coeff in zip(op.operator.paulis, op.operator.coeffs)}
    for edge, (pauli, coeff) in required_paulis.items():
        qubits = [dag.qubits[edge[0]], dag.qubits[edge[1]]]
        simple_pauli = Pauli(pauli.to_label().replace('I', ''))
        pauli_2q = PauliEvolutionGate(simple_pauli, op.time * np.real(coeff))
        sub_dag.apply_operation_back(pauli_2q, qubits)
    return sub_dag