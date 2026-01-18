from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from .gate_sequence import GateSequence
@optionals.HAS_SKLEARN.require_in_call
def _check_candidate_kdtree(candidate, existing_sequences, tol=1e-10):
    """Check if there's a candidate implementing the same matrix up to ``tol``.

    This uses a k-d tree search and is much faster than the greedy, list-based search.
    """
    from sklearn.neighbors import KDTree
    if any((candidate.name == existing.name for existing in existing_sequences)):
        return False
    points = np.array([sequence.product.flatten() for sequence in existing_sequences])
    candidate = np.array([candidate.product.flatten()])
    kdtree = KDTree(points)
    dist, _ = kdtree.query(candidate)
    return dist[0][0] > tol