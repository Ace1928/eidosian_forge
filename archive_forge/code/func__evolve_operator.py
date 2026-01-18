from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.synthesis.evolution import LieTrotter
from .n_local import NLocal
def _evolve_operator(self, operator, time):
    from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate
    if isinstance(operator, Operator):
        gate = HamiltonianGate(operator, time)
    else:
        evolution = LieTrotter() if self._evolution is None else self._evolution
        gate = PauliEvolutionGate(operator, time, synthesis=evolution)
    evolved = QuantumCircuit(operator.num_qubits)
    if not self.flatten:
        evolved.append(gate, evolved.qubits)
    else:
        evolved.compose(gate.definition, evolved.qubits, inplace=True)
    return evolved