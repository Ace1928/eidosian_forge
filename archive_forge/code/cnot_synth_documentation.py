from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError

    This function is a helper function of the algorithm for optimal synthesis
    of linear reversible circuits (the Patel–Markov–Hayes algorithm). It works
    like gaussian elimination, except that it works a lot faster, and requires
    fewer steps (and therefore fewer CNOTs). It takes the matrix "state" and
    splits it into sections of size section_size. Then it eliminates all non-zero
    sub-rows within each section, which are the same as a non-zero sub-row
    above. Once this has been done, it continues with normal gaussian elimination.
    The benefit is that with small section sizes (m), most of the sub-rows will
    be cleared in the first step, resulting in a factor m fewer row row operations
    during Gaussian elimination.

    The algorithm is described in detail in the following paper
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Note:
    This implementation tweaks the Patel, Markov, and Hayes algorithm by adding
    a "back reduce" which adds rows below the pivot row with a high degree of
    overlap back to it. The intuition is to avoid a high-weight pivot row
    increasing the weight of lower rows.

    Args:
        state (ndarray): n x n matrix, describing a linear quantum circuit
        section_size (int): the section size the matrix columns are divided into

    Returns:
        numpy.matrix: n by n matrix, describing the state of the output circuit
        list: a k by 2 list of C-NOT operations that need to be applied
    