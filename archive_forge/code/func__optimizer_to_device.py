from typing import Iterable
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.optim import Optimizer
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.types import _DEVICE
def _optimizer_to_device(optimizer: Optimizer, device: _DEVICE) -> None:
    """Moves the state of a single optimizer to the device."""
    for p, v in optimizer.state.items():
        optimizer.state[p] = apply_to_collection(v, Tensor, move_data_to_device, device, allow_frozen=True)