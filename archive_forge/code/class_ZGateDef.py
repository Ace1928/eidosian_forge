from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
class ZGateDef(cirq.EigenGate, cirq.testing.TwoQubitGate):

    @property
    def exponent(self):
        return self._exponent

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 0])), (1, np.diag([0, 1]))]