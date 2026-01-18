import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class InconsistentDecompostion(cirq.Operation):

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(rotations=1)

    def _decompose_(self) -> cirq.OP_TREE:
        yield cirq.X(self.qubits[0])

    @property
    def qubits(self):
        return tuple(cirq.LineQubit(3).range(3))

    def with_qubits(self, _):
        pass