from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def _create_device(qubits: Iterable[cirq.GridQubit]):
    return FakeDevice(qubits)