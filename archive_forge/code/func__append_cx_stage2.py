import numpy as np
from qiskit.circuit import QuantumCircuit
def _append_cx_stage2(qc, n):
    """A single layer of CX gates."""
    for i in range(n // 2):
        qc.cx(2 * i + 1, 2 * i)
    for i in range((n + 1) // 2 - 1):
        qc.cx(2 * i + 1, 2 * i + 2)
    return qc