from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import STEP_OUTPUT
class RandomIterableDatasetWithLen(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self) -> int:
        return self.count