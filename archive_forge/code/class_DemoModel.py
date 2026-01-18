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
class DemoModel(LightningModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, out_dim: int=10, learning_rate: float=0.02):
        super().__init__()
        self.l1 = torch.nn.Linear(32, out_dim)
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch: Any, batch_nb: int) -> STEP_OUTPUT:
        x = batch
        x = self(x)
        return x.sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)