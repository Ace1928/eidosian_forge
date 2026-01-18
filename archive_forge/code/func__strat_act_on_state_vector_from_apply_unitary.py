from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
def _strat_act_on_state_vector_from_apply_unitary(action: Any, args: 'cirq.StateVectorSimulationState', qubits: Sequence['cirq.Qid']) -> bool:
    if not args._state.apply_unitary(action, args.get_axes(qubits)):
        return NotImplemented
    return True