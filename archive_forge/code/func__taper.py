from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
def _taper(self, op: SparsePauliOp, curr_tapering_values: list[int]) -> SparsePauliOp:
    pauli_list = []
    for pauli_term in iter(op):
        coeff_out = pauli_term.coeffs[0]
        for idx, qubit_idx in enumerate(self._sq_list):
            if pauli_term.paulis.z[0, qubit_idx] or pauli_term.paulis.x[0, qubit_idx]:
                coeff_out = curr_tapering_values[idx] * coeff_out
        z_temp = np.delete(pauli_term.paulis.z[0].copy(), np.asarray(self._sq_list))
        x_temp = np.delete(pauli_term.paulis.x[0].copy(), np.asarray(self._sq_list))
        pauli_list.append((Pauli((z_temp, x_temp)).to_label(), coeff_out))
    spo = SparsePauliOp.from_list(pauli_list).simplify(atol=0.0)
    spo = spo.chop(self.tol)
    return spo