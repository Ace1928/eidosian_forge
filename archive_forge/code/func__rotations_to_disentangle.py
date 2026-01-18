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
def _rotations_to_disentangle(local_param):
    """
        Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.

        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
        [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
        0         0           Ry(theta_2^n).Rz(phi_2^n)]]
        """
    remaining_vector = []
    thetas = []
    phis = []
    param_len = len(local_param)
    for i in range(param_len // 2):
        remains, add_theta, add_phi = StatePreparation._bloch_angles(local_param[2 * i:2 * (i + 1)])
        remaining_vector.append(remains)
        thetas.append(-add_theta)
        phis.append(-add_phi)
    return (remaining_vector, thetas, phis)