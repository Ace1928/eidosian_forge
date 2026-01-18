import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
    """Delegates the call to measure the density matrix."""
    if self._state is not None:
        return self._state.measure(self.get_axes(qubits), self.prng)
    raise NotImplementedError()