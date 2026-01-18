from typing import Union, Optional
import math
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.states.statevector import Statevector  # pylint: disable=cyclic-import
def _gates_to_uncompute(self):
    """Call to create a circuit with gates that take the desired vector to zero.

        Returns:
            QuantumCircuit: circuit to take self.params vector to :math:`|{00\\ldots0}\\rangle`
        """
    q = QuantumRegister(self.num_qubits)
    circuit = QuantumCircuit(q, name='disentangler')
    remaining_param = self.params
    for i in range(self.num_qubits):
        remaining_param, thetas, phis = StatePreparation._rotations_to_disentangle(remaining_param)
        add_last_cnot = True
        if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
            add_last_cnot = False
        if np.linalg.norm(phis) != 0:
            rz_mult = self._multiplex(RZGate, phis, last_cnot=add_last_cnot)
            circuit.append(rz_mult.to_instruction(), q[i:self.num_qubits])
        if np.linalg.norm(thetas) != 0:
            ry_mult = self._multiplex(RYGate, thetas, last_cnot=add_last_cnot)
            circuit.append(ry_mult.to_instruction().reverse_ops(), q[i:self.num_qubits])
    circuit.global_phase -= np.angle(sum(remaining_param))
    return circuit