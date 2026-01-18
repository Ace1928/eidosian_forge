from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
def _apply_cu(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        circuit.p((lam + phi) / 2, [control])
        circuit.p((lam - phi) / 2, [target])
        circuit.cx(control, target)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, [target])
        circuit.cx(control, target)
        circuit.u(theta / 2, phi, 0, [target])
    else:
        circuit.cu(theta, phi, lam, 0, control, target)