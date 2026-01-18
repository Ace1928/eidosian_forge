import cirq
import numpy as np
import pytest
from cirq_ft.infra.decompose_protocol import (
class DecomposeEmptyList(cirq.testing.SingleQubitGate):

    def _decompose_(self, _):
        return []