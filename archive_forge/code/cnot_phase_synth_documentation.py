from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.synthesis.linear import synth_cnot_count_full_pmh

    Remove duplicates in list

    Args:
        lists (list): a list which may contain duplicate elements.

    Returns:
        list: a list which contains only unique elements.
    