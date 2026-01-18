import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import CNOTDihedral
Decompose a :class:`.CNOTDihedral` element on a single qubit and two
    qubits into a :class:`.QuantumCircuit`.
    This decomposition has an optimal number of :class:`.CXGate`\ s.

    Args:
        elem: A :class:`.CNOTDihedral` element.

    Returns:
        A circuit implementation of the :class:`.CNOTDihedral` element.

    Raises:
        QiskitError: if the element in not 1-qubit or 2-qubit :class:`.CNOTDihedral`.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
    