import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class G2(cirq.Gate):

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(self, qubits, context):
        yield G1()(*context.qubit_manager.qalloc(1))