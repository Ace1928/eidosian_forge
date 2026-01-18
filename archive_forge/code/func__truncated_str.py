from __future__ import annotations
from collections import defaultdict
from typing import Literal
import numpy as np
import rustworkx as rx
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
def _truncated_str(self, show_class):
    stop = self._num_paulis
    if self.__truncate__ and self.num_qubits > 0:
        max_paulis = self.__truncate__ // self.num_qubits
        if self._num_paulis > max_paulis:
            stop = max_paulis
    labels = [str(self[i]) for i in range(stop)]
    prefix = 'PauliList(' if show_class else ''
    tail = ')' if show_class else ''
    if stop != self._num_paulis:
        suffix = ', ...]' + tail
    else:
        suffix = ']' + tail
    list_str = np.array2string(np.array(labels), threshold=stop + 1, separator=', ', prefix=prefix, suffix=suffix)
    return prefix + list_str[:-1] + suffix