from typing import Callable, Optional, Union, Any, Dict
from functools import partial
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from .evolution_synthesis import EvolutionSynthesis
def _default_atomic_evolution(operator, time, cx_structure):
    if isinstance(operator, Pauli):
        evolution_circuit = evolve_pauli(operator, time, cx_structure)
    else:
        pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operator.to_list()]
        name = f'exp(it {[pauli.to_label() for pauli, _ in pauli_list]})'
        evolution_circuit = QuantumCircuit(operator.num_qubits, name=name)
        for pauli, coeff in pauli_list:
            evolution_circuit.compose(evolve_pauli(pauli, coeff * time, cx_structure), inplace=True)
    return evolution_circuit