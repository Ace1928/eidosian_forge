from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from .gate_sequence import GateSequence
def _check_candidate_greedy(candidate, existing_sequences, tol=1e-10):
    if any((candidate.name == existing.name for existing in existing_sequences)):
        return False
    for existing in existing_sequences:
        if matrix_equal(existing.product_su2, candidate.product_su2, ignore_phase=True, atol=tol):
            return len(candidate.gates) < len(existing.gates)
    return True