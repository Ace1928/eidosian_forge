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
def _define_from_int(self):
    q = QuantumRegister(self.num_qubits, 'q')
    initialize_circuit = QuantumCircuit(q, name='init_def')
    intstr = f'{int(np.real(self.params[0])):0{self.num_qubits}b}'[::-1]
    if len(intstr) > self.num_qubits:
        raise QiskitError('StatePreparation integer has %s bits, but this exceeds the number of qubits in the circuit, %s.' % (len(intstr), self.num_qubits))
    for qubit, bit in enumerate(intstr):
        if bit == '1':
            initialize_circuit.append(XGate(), [q[qubit]])
    return initialize_circuit