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
def _paulis2inds(paulis: PauliList) -> list[int]:
    """Convert PauliList to diagonal integers.
    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    nonid = paulis.z | paulis.x
    inds = [0] * paulis.size
    packed_vals = np.packbits(nonid, axis=1, bitorder='little')
    for i, vals in enumerate(packed_vals):
        for j, val in enumerate(vals):
            inds[i] += val.item() * (1 << 8 * j)
    return inds