import numpy as np
from qiskit.circuit import QuantumCircuit
Synthesis of a CZ circuit for linear nearest neighbour (LNN) connectivity,
    based on Maslov and Roetteler.

    Note that this method *reverts* the order of qubits in the circuit,
    and returns a circuit containing :class:`.CXGate`\s and phase gates
    (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).

    Args:
        mat: an upper-diagonal matrix representing the CZ circuit.
            ``mat[i][j]=1 for i<j`` represents a ``cz(i,j)`` gate

    Returns:
        A circuit implementation of the CZ circuit of depth :math:`2n+2` for LNN
        connectivity.

    References:
        1. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    