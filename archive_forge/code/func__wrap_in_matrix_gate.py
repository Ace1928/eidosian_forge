from typing import List
import numpy as np
import pytest
import cirq
def _wrap_in_matrix_gate(ops: cirq.OP_TREE):
    op = _wrap_in_cop(ops, 'temp')
    return cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits)