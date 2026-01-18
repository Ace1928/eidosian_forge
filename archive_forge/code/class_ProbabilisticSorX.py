from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
class ProbabilisticSorX(cirq.Gate):

    def num_qubits(self) -> int:
        return 1

    def _kraus_(self):
        return [cirq.unitary(cirq.S) * np.sqrt(1 / 3), cirq.unitary(cirq.X) * np.sqrt(2 / 3)]