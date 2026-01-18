from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def _overrotation(op):
    if isinstance(op.gate, cirq.XPowGate):
        return cirq.XPowGate(exponent=op.gate.exponent + 0.1).on(*op.qubits)
    return op