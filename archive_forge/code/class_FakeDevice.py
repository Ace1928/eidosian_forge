from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
class FakeDevice(cirq.Device):

    def __init__(self, qubits):
        self.qubits = qubits

    @property
    def metadata(self):
        return cirq.DeviceMetadata(self.qubits, None)