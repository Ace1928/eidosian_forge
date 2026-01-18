import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
class MPSTrialResult(simulator_base.SimulationTrialResultBase['MPSState']):
    """A single trial reult"""

    def __init__(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[MPSState]') -> None:
        super().__init__(params=params, measurements=measurements, final_simulator_state=final_simulator_state)

    @property
    def final_state(self) -> 'MPSState':
        return self._get_merged_sim_state()

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._final_simulator_state
        return f'measurements: {samples}\noutput state: {final}'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        if cycle:
            p.text('cirq.MPSTrialResult(...)')
        else:
            p.text(str(self))