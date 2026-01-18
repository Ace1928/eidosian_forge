from __future__ import annotations
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
def _dec_ucrot(self):
    """
        Finds a decomposition of a UC rotation gate into elementary gates
        (C-NOTs and single-qubit rotations).
        """
    q = QuantumRegister(self.num_qubits)
    circuit = QuantumCircuit(q)
    q_target = q[0]
    q_controls = q[1:]
    if not q_controls:
        if self.rot_axes == 'X':
            if np.abs(self.params[0]) > _EPS:
                circuit.rx(self.params[0], q_target)
        if self.rot_axes == 'Y':
            if np.abs(self.params[0]) > _EPS:
                circuit.ry(self.params[0], q_target)
        if self.rot_axes == 'Z':
            if np.abs(self.params[0]) > _EPS:
                circuit.rz(self.params[0], q_target)
    else:
        angles = self.params.copy()
        UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
        for i, angle in enumerate(angles):
            if self.rot_axes == 'X':
                if np.abs(angle) > _EPS:
                    circuit.rx(angle, q_target)
            if self.rot_axes == 'Y':
                if np.abs(angle) > _EPS:
                    circuit.ry(angle, q_target)
            if self.rot_axes == 'Z':
                if np.abs(angle) > _EPS:
                    circuit.rz(angle, q_target)
            if not i == len(angles) - 1:
                binary_rep = np.binary_repr(i + 1)
                q_contr_index = len(binary_rep) - len(binary_rep.rstrip('0'))
            else:
                q_contr_index = len(q_controls) - 1
            if self.rot_axes == 'X':
                circuit.ry(np.pi / 2, q_target)
            circuit.cx(q_controls[q_contr_index], q_target)
            if self.rot_axes == 'X':
                circuit.ry(-np.pi / 2, q_target)
    return circuit