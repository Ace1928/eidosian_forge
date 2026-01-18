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
class FakeSimulatesSamples(SimulatesSamples):
    """A SimulatesSamples that returns specified values from _run."""

    def __init__(self, run_output: Dict[str, np.ndarray]):
        self._run_output = run_output

    def _run(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        return self._run_output