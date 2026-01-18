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
@staticmethod
def _from_paulis(data):
    """Construct a PauliList from a list of Pauli data.

        Args:
            data (iterable): list of Pauli data.

        Returns:
            PauliList: the constructed PauliList.

        Raises:
            QiskitError: If the input list is empty or contains invalid
            Pauli strings.
        """
    if not isinstance(data, (list, tuple, set, np.ndarray)):
        data = [data]
    num_paulis = len(data)
    if num_paulis == 0:
        raise QiskitError('Input Pauli list is empty.')
    paulis = []
    for i in data:
        if not isinstance(i, Pauli):
            paulis.append(Pauli(i))
        else:
            paulis.append(i)
    num_qubits = paulis[0].num_qubits
    base_z = np.zeros((num_paulis, num_qubits), dtype=bool)
    base_x = np.zeros((num_paulis, num_qubits), dtype=bool)
    base_phase = np.zeros(num_paulis, dtype=int)
    for i, pauli in enumerate(paulis):
        if pauli.num_qubits != num_qubits:
            raise ValueError(f'The {i}th Pauli is defined over {pauli.num_qubits} qubits, but num_qubits == {num_qubits} was expected.')
        base_z[i] = pauli._z
        base_x[i] = pauli._x
        base_phase[i] = pauli._phase.item()
    return (base_z, base_x, base_phase)