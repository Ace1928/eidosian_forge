import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class IsCachable(cirq.Operation):

    def __init__(self) -> None:
        super().__init__()
        self.num_calls = 0
        self._gate = cirq.X

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        self.num_calls += 1
        return cirq_ft.TComplexity()

    @property
    def qubits(self):
        return [cirq.LineQubit(3)]

    def with_qubits(self, _):
        ...

    @property
    def gate(self):
        return self._gate