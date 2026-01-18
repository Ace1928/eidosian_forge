import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def _assert_not_mirror(gate) -> None:
    trans_x = gate.pauli_tuple(cirq.X)
    trans_y = gate.pauli_tuple(cirq.Y)
    trans_z = gate.pauli_tuple(cirq.Z)
    right_handed = trans_x[1] ^ trans_y[1] ^ trans_z[1] ^ (trans_x[0].relative_index(trans_y[0]) != 1)
    assert right_handed, 'Mirrors'