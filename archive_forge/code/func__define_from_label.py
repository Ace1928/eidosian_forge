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
def _define_from_label(self):
    q = QuantumRegister(self.num_qubits, 'q')
    initialize_circuit = QuantumCircuit(q, name='init_def')
    for qubit, param in enumerate(reversed(self.params)):
        if param == '1':
            initialize_circuit.append(XGate(), [q[qubit]])
        elif param == '+':
            initialize_circuit.append(HGate(), [q[qubit]])
        elif param == '-':
            initialize_circuit.append(XGate(), [q[qubit]])
            initialize_circuit.append(HGate(), [q[qubit]])
        elif param == 'r':
            initialize_circuit.append(HGate(), [q[qubit]])
            initialize_circuit.append(SGate(), [q[qubit]])
        elif param == 'l':
            initialize_circuit.append(HGate(), [q[qubit]])
            initialize_circuit.append(SdgGate(), [q[qubit]])
    if self._inverse:
        initialize_circuit = initialize_circuit.inverse()
    return initialize_circuit