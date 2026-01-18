from abc import ABC
from functools import partial
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from lightning_fabric.utilities.types import _DEVICE
def _from_numpy(value: np.ndarray, device: _DEVICE) -> Tensor:
    return torch.from_numpy(value).to(device)