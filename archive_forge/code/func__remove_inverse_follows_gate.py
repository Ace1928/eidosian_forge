from __future__ import annotations
import numpy as np
from qiskit.circuit.gate import Gate
from .gate_sequence import GateSequence
from .commutator_decompose import commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses
def _remove_inverse_follows_gate(sequence):
    index = 0
    while index < len(sequence.gates) - 1:
        curr_gate = sequence.gates[index]
        next_gate = sequence.gates[index + 1]
        if curr_gate.name in _1q_inverses.keys():
            remove = _1q_inverses[curr_gate.name] == next_gate.name
        else:
            remove = curr_gate.inverse() == next_gate
        if remove:
            sequence.remove_cancelling_pair([index, index + 1])
            if index > 0:
                index -= 1
        else:
            index += 1