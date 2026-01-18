from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _decompose_clifford_1q(tableau):
    """Decompose a single-qubit clifford"""
    circuit = QuantumCircuit(1, name='temp')
    destab_phase, stab_phase = tableau[:, 2]
    if destab_phase and (not stab_phase):
        circuit.z(0)
    elif not destab_phase and stab_phase:
        circuit.x(0)
    elif destab_phase and stab_phase:
        circuit.y(0)
    destab_phase_label = '-' if destab_phase else '+'
    stab_phase_label = '-' if stab_phase else '+'
    destab_x, destab_z = (tableau[0, 0], tableau[0, 1])
    stab_x, stab_z = (tableau[1, 0], tableau[1, 1])
    if stab_z and (not stab_x):
        stab_label = 'Z'
        if destab_z:
            destab_label = 'Y'
            circuit.s(0)
        else:
            destab_label = 'X'
    elif not stab_z and stab_x:
        stab_label = 'X'
        if destab_x:
            destab_label = 'Y'
            circuit.sdg(0)
        else:
            destab_label = 'Z'
        circuit.h(0)
    else:
        stab_label = 'Y'
        if destab_z:
            destab_label = 'Z'
        else:
            destab_label = 'X'
            circuit.s(0)
        circuit.h(0)
        circuit.s(0)
    name_destab = f"Destabilizer = ['{destab_phase_label}{destab_label}']"
    name_stab = f"Stabilizer = ['{stab_phase_label}{stab_label}']"
    circuit.name = f'Clifford: {name_stab}, {name_destab}'
    return circuit