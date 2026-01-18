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
class ManualOptimBoringModel(BoringModel):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        opt = self.optimizers()
        assert isinstance(opt, (Optimizer, LightningOptimizer))
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss