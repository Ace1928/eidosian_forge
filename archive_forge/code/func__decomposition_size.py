import numpy as np
import scipy.stats
import cirq
def _decomposition_size(U, controls_count):
    qubits = cirq.LineQubit.range(controls_count + 1)
    operations = cirq.decompose_multi_controlled_rotation(U, qubits[:controls_count], qubits[-1])
    return _count_operations(operations)