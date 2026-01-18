from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
class NoiseModelWithNoisyMomentMethod(cirq.NoiseModel):

    def noisy_moment(self, moment, system_qubits):
        return [y.with_tags(ops.VirtualTag()) for y in cirq.Y.on_each(*moment.qubits)]