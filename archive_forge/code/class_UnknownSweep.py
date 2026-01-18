from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
class UnknownSweep(sweeps.SingleSweep):

    def _tuple(self):
        return (self.key, tuple(range(10)))

    def __len__(self) -> int:
        return 10

    def _values(self) -> Iterator[float]:
        return iter(range(10))