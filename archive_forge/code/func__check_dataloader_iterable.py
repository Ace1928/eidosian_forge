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
def _check_dataloader_iterable(dataloader: object, source: _DataLoaderSource, trainer_fn: TrainerFn) -> None:
    if isinstance(dataloader, DataLoader):
        return
    try:
        iter(dataloader)
    except TypeError:
        prefix = 'train_' if trainer_fn == TrainerFn.FITTING else ''
        if not source.is_module():
            raise TypeError(f'An invalid dataloader was passed to `Trainer.{trainer_fn.value}({prefix}dataloaders=...)`. Found {dataloader}.')
        if not is_overridden(source.name, source.instance):
            raise TypeError(f'An invalid dataloader was passed to `Trainer.{trainer_fn.value}({prefix}dataloaders=...)`. Found {dataloader}. Either pass the dataloader to the `.{trainer_fn.value}()` method OR implement `def {source.name}(self):` in your LightningModule/LightningDataModule.')
        raise TypeError(f'An invalid dataloader was returned from `{type(source.instance).__name__}.{source.name}()`. Found {dataloader}.')