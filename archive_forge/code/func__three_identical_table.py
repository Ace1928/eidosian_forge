import numpy as np
import pytest
import cirq
def _three_identical_table(num_qubits):
    t1 = cirq.CliffordTableau(num_qubits)
    t2 = cirq.CliffordTableau(num_qubits)
    t3 = cirq.CliffordTableau(num_qubits)
    return (t1, t2, t3)