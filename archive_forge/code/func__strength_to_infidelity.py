from __future__ import annotations
import heapq
import math
from operator import itemgetter
from typing import Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXXGate, RZXGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.one_qubit.one_qubit_decompose import ONE_QUBIT_EULER_BASIS_GATES
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from .utilities import EPSILON
from .polytopes import XXPolytope
@staticmethod
def _strength_to_infidelity(basis_fidelity, approximate=False):
    """
        Converts a dictionary mapping XX strengths to fidelities to a dictionary mapping XX
        strengths to infidelities. Also supports one of the other formats Qiskit uses: if only a
        lone float is supplied, it extends it from CX over CX/2 and CX/3 by linear decay.
        """
    if isinstance(basis_fidelity, float):
        if not approximate:
            slope, offset = (1e-10, 1e-12)
        else:
            slope, offset = ((1 - basis_fidelity) / 2, (1 - basis_fidelity) / 2)
        return {strength: slope * strength / (np.pi / 2) + offset for strength in [np.pi / 2, np.pi / 4, np.pi / 6]}
    elif isinstance(basis_fidelity, dict):
        return {strength: 1 - fidelity if approximate else 1e-12 + 1e-10 * strength / (np.pi / 2) for strength, fidelity in basis_fidelity.items()}
    raise TypeError('Unknown basis_fidelity payload.')