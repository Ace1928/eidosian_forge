import numpy as np
import pytest
import cirq
class GoodGateNoPauliExpansion(cirq.Gate):

    def num_qubits(self) -> int:
        return 4