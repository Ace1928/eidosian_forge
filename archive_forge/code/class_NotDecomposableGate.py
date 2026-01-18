import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class NotDecomposableGate(cirq.Gate):

    def num_qubits(self):
        return 1