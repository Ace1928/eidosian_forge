from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, OrderedDict
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.optimization.closure import AbstractClosure, OutputResult
from pytorch_lightning.loops.progress import _OptimizationProgress
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _optimizer_zero_grad(self, batch_idx: int, optimizer: torch.optim.Optimizer) -> None:
    """Zeroes out all gradients of parameters optimized by the current optimizer.

        Args:
            batch_idx: the index of the current batch
            optimizer: the current optimizer

        """
    trainer = self.trainer
    call._call_lightning_module_hook(trainer, 'optimizer_zero_grad', trainer.current_epoch, batch_idx, optimizer)
    self.optim_progress.optimizer.zero_grad.increment_completed()