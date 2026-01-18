import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock
import duet
import numpy as np
import pytest
import cirq
from cirq import study
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulator import (
class Reset11To00(cirq.Gate):

    def num_qubits(self) -> int:
        return 2

    def _kraus_(self):
        return [np.eye(4) - cirq.one_hot(index=(3, 3), shape=(4, 4), dtype=np.complex64), cirq.one_hot(index=(0, 3), shape=(4, 4), dtype=np.complex64)]