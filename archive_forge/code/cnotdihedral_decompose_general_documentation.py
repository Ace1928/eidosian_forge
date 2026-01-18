import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import CNOTDihedral
Decompose a :class:`.CNOTDihedral` element into a :class:`.QuantumCircuit`.

    Decompose a general :class:`.CNOTDihedral` elements.
    The number of CX gates is not necessarily optimal.
    For a decomposition of a 1-qubit or 2-qubit element, call
    :func:`.synth_cnotdihedral_two_qubits`.

    Args:
        elem: A :class:`.CNOTDihedral` element.

    Returns:
        A circuit implementation of the :class:`.CNOTDihedral` element.

    Raises:
        QiskitError: if the element could not be decomposed into a circuit.

    References:
        1. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    