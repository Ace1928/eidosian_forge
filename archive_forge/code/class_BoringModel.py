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
class BoringModel(LightningModule):
    """Testing PL Module.

    Use as follows:
    - subclass
    - modify the behavior for what you want

    .. warning::  This is meant for testing/debugging and is experimental.

    Example::

        class TestModel(BoringModel):
            def training_step(self, ...):
                ...  # do your own thing

    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def loss(self, preds: Tensor, labels: Optional[Tensor]=None) -> Tensor:
        if labels is None:
            labels = torch.ones_like(preds)
        return torch.nn.functional.mse_loss(preds, labels)

    def step(self, batch: Any) -> Tensor:
        output = self(batch)
        return self.loss(output)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return {'loss': self.step(batch)}

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return {'x': self.step(batch)}

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return {'y': self.step(batch)}

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[_TORCH_LRSCHEDULER]]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return ([optimizer], [lr_scheduler])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))