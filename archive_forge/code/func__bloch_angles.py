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
@staticmethod
def _bloch_angles(pair_of_complex):
    """
        Static internal method to work out rotation to create the passed-in
        qubit from the zero vector.
        """
    [a_complex, b_complex] = pair_of_complex
    a_complex = complex(a_complex)
    b_complex = complex(b_complex)
    mag_a = abs(a_complex)
    final_r = np.sqrt(mag_a ** 2 + np.absolute(b_complex) ** 2)
    if final_r < _EPS:
        theta = 0
        phi = 0
        final_r = 0
        final_t = 0
    else:
        theta = 2 * np.arccos(mag_a / final_r)
        a_arg = np.angle(a_complex)
        b_arg = np.angle(b_complex)
        final_t = a_arg + b_arg
        phi = b_arg - a_arg
    return (final_r * np.exp(1j * final_t / 2), theta, phi)