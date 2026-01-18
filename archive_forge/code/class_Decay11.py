from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
class Decay11(cirq.Gate):

    def num_qubits(self) -> int:
        return 2

    def _kraus_(self):
        bottom_right = cirq.one_hot(index=(3, 3), shape=(4, 4), dtype=np.complex64)
        top_right = cirq.one_hot(index=(0, 3), shape=(4, 4), dtype=np.complex64)
        return [np.eye(4) * np.sqrt(3 / 4), (np.eye(4) - bottom_right) * np.sqrt(1 / 4), top_right * np.sqrt(1 / 4)]