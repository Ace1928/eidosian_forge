from typing import List, TYPE_CHECKING
import functools
import numpy as np
from cirq import ops, protocols, qis, sim
Decompose an n-qubit Clifford Tableau into a list of one/two qubit operations.

    The implementation is based on Theorem 8 in [1].
    [1] S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
        Phys. Rev. A 70, 052328 (2004). https://arxiv.org/abs/quant-ph/0406196

    Args:
        qubits: The list of qubits being operated on.
        clifford_tableau: The Clifford Tableau for decomposition.

    Returns:
        A list of operations reconstructs the same Clifford tableau.

    Raises:
        ValueError: The length of input qubit mismatch with the size of tableau.
    