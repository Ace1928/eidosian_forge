from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def _map_indices_apply_operation(**kwargs):
    """Map indices to wires.

    Args:
        **kwargs (dict): Stores indices calculated in `get_einsum_mapping`:
            state_indices (str): Indices that are summed.
            row_indices (str): Indices that must be replaced with sums.
            new_row_indices (str): Tensor indices of the state.

    Returns:
        String of einsum indices to complete einsum calculations.
    """
    op_1_indices = f'{kwargs['new_row_indices']}{kwargs['row_indices']}'
    new_state_indices = get_new_state_einsum_indices(old_indices=kwargs['row_indices'], new_indices=kwargs['new_row_indices'], state_indices=kwargs['state_indices'])
    return f'{op_1_indices},...{kwargs['state_indices']}->...{new_state_indices}'