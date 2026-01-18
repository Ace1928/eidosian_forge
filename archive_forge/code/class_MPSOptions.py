import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
@dataclasses.dataclass(frozen=True)
class MPSOptions:
    method: str = 'svds'
    max_bond: Optional[int] = None
    cutoff_mode: str = 'rsum2'
    cutoff: float = 1e-06
    sum_prob_atol: float = 0.001