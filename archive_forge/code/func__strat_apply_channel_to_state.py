from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
def _strat_apply_channel_to_state(action: Any, args: 'cirq.DensityMatrixSimulationState', qubits: Sequence['cirq.Qid']) -> bool:
    """Apply channel to state."""
    if not args._state.apply_channel(action, args.get_axes(qubits)):
        return NotImplemented
    return True