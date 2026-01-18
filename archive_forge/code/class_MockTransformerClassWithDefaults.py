from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@cirq.transformer
class MockTransformerClassWithDefaults:

    def __init__(self):
        self.mock = mock.Mock()

    def __call__(self, circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext]=cirq.TransformerContext(), atol: float=0.0001, custom_arg: CustomArg=CustomArg()) -> cirq.AbstractCircuit:
        self.mock(circuit, context, atol, custom_arg)
        return circuit[::-1]