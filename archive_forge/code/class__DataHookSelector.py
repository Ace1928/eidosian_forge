import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple, Union
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from lightning_fabric.utilities.data import (
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from pytorch_lightning.overrides.distributed import UnrepeatedDistributedSamplerWrapper
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import _is_dataloader_shuffled, _update_dataloader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities.warnings import PossibleUserWarning
@dataclass
class _DataHookSelector:
    """Stores the info about the shared DataHooks within ``LightningModule`` and ``LightningDataModule``.

    The hook source can be:

    1. the :class:`~pytorch_lightning.core.LightningModule`,
    2. the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`,

    Arguments:
        model: A ``LightningModule``
        datamodule: A ``LightningDataModule``

    """
    model: 'pl.LightningModule'
    datamodule: Optional['pl.LightningDataModule']
    _valid_hooks: Tuple[str, ...] = field(default=('on_before_batch_transfer', 'transfer_batch_to_device', 'on_after_batch_transfer'))

    def get_instance(self, hook_name: str) -> Union['pl.LightningModule', 'pl.LightningDataModule']:
        if hook_name not in self._valid_hooks:
            raise ValueError(f'`{hook_name}` is not a shared hook within `LightningModule` and `LightningDataModule`. Valid hooks are {self._valid_hooks}.')
        if self.datamodule is None:
            return self.model
        if is_overridden(hook_name, self.datamodule):
            if is_overridden(hook_name, self.model):
                warning_cache.warn(f'You have overridden `{hook_name}` in both `LightningModule` and `LightningDataModule`. It will use the implementation from `LightningDataModule` instance.')
            return self.datamodule
        if is_overridden(hook_name, self.model):
            warning_cache.warn(f'You have overridden `{hook_name}` in `LightningModule` but have passed in a `LightningDataModule`. It will use the implementation from `LightningModule` instance.')
        return self.model