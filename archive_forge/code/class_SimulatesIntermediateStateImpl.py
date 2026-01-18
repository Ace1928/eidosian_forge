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
class SimulatesIntermediateStateImpl(Generic[TStepResult, TSimulationState], SimulatesIntermediateState[TStepResult, 'SimulationTrialResult', TSimulationState], metaclass=abc.ABCMeta):
    """A SimulatesIntermediateState that uses the default SimulationTrialResult type."""

    def _create_simulator_trial_result(self, params: study.ParamResolver, measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[TSimulationState]') -> 'SimulationTrialResult':
        """This method creates a default trial result.

        Args:
            params: The ParamResolver for this trial.
            measurements: The measurement results for this trial.
            final_simulator_state: The final state of the simulation.

        Returns:
            The SimulationTrialResult.
        """
        return SimulationTrialResult(params=params, measurements=measurements, final_simulator_state=final_simulator_state)