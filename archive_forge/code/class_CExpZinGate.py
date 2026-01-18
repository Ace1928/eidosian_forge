from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
class CExpZinGate(cirq.EigenGate, cirq.testing.TwoQubitGate):
    """Two-qubit gate for the following matrix:
    [1  0  0  0]
    [0  1  0  0]
    [0  0  i  0]
    [0  0  0 -i]
    """

    def __init__(self, quarter_turns: value.TParamVal) -> None:
        super().__init__(exponent=quarter_turns)

    @property
    def exponent(self):
        return self._exponent

    def _with_exponent(self, exponent):
        return CExpZinGate(exponent)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 1, 0, 0])), (0.5, np.diag([0, 0, 1, 0])), (-0.5, np.diag([0, 0, 0, 1]))]