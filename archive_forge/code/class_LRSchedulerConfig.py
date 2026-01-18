from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
import torch
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import Metric
from typing_extensions import NotRequired, Required
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER, LRScheduler, ProcessGroup, ReduceLROnPlateau
@dataclass
class LRSchedulerConfig:
    scheduler: Union[_TORCH_LRSCHEDULER, ReduceLROnPlateau]
    name: Optional[str] = None
    interval: str = 'epoch'
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: Optional[str] = None
    strict: bool = True