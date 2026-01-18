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
def _define_synthesis(self):
    """Calculate a subcircuit that implements this initialization

        Implements a recursive initialization algorithm, including optimizations,
        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
        https://arxiv.org/abs/quant-ph/0406176v5

        Additionally implements some extra optimizations: remove zero rotations and
        double cnots.
        """
    disentangling_circuit = self._gates_to_uncompute()
    if self._inverse is False:
        initialize_instr = disentangling_circuit.to_instruction().inverse()
    else:
        initialize_instr = disentangling_circuit.to_instruction()
    q = QuantumRegister(self.num_qubits, 'q')
    initialize_circuit = QuantumCircuit(q, name='init_def')
    initialize_circuit.append(initialize_instr, q[:])
    return initialize_circuit