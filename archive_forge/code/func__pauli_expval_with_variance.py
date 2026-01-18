from __future__ import annotations
from collections.abc import Sequence
from itertools import accumulate
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1, BackendV2, Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key, _observable_key, init_observable
def _pauli_expval_with_variance(counts: Counts, paulis: PauliList) -> tuple[np.ndarray, np.ndarray]:
    """Return array of expval and variance pairs for input Paulis.
    Note: All non-identity Pauli's are treated as Z-paulis, assuming
    that basis rotations have been applied to convert them to the
    diagonal basis.
    """
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)
    expvals = np.zeros(size, dtype=float)
    denom = 0
    for bin_outcome, freq in counts.items():
        split_outcome = bin_outcome.split(' ', 1)[0] if ' ' in bin_outcome else bin_outcome
        outcome = int(split_outcome, 2)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff
    expvals /= denom
    variances = 1 - expvals ** 2
    return (expvals, variances)