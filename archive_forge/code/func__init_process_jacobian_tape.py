from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def _init_process_jacobian_tape(self, tape, starting_state, use_device_state):
    """Generate an initial state vector for ``_process_jacobian_tape``."""
    if starting_state is not None:
        if starting_state.size != 2 ** len(self.wires):
            raise QuantumFunctionError('The number of qubits of starting_state must be the same as that of the device.')
        self._apply_state_vector(starting_state, self.wires)
    elif not use_device_state:
        self.reset()
        self.apply(tape.operations)
    return self.state_vector