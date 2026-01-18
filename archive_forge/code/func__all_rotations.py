import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def _all_rotations():
    for pauli, flip in itertools.product(_paulis, _bools):
        yield (pauli, flip)