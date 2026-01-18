from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def _apply_basis_state(self, state, wires):
    """Initialize the state vector in a specified computational basis state.

            Args:
                state (array[int]): computational basis state of shape ``(wires,)``
                    consisting of 0s and 1s.
                wires (Wires): wires that the provided computational state should be
                    initialized on

            Note: This function does not support broadcasted inputs yet.
            """
    num = self._get_basis_state_index(state, wires)
    self._create_basis_state(num)