from __future__ import annotations
import numpy as np
from qiskit.circuit.gate import Gate
from .gate_sequence import GateSequence
from .commutator_decompose import commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses
def _remove_identities(sequence):
    index = 0
    while index < len(sequence.gates):
        if sequence.gates[index].name == 'id':
            sequence.gates.pop(index)
        else:
            index += 1