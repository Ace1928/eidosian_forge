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
def _worker_check(trainer: 'pl.Trainer', dataloader: object, name: str) -> None:
    if not isinstance(dataloader, DataLoader):
        return
    upper_bound = suggested_max_num_workers(trainer.num_devices)
    start_method = dataloader.multiprocessing_context.get_start_method() if dataloader.multiprocessing_context is not None else mp.get_start_method()
    if dataloader.num_workers > 0 and start_method == 'spawn' and (not dataloader.persistent_workers):
        rank_zero_warn(f"Consider setting `persistent_workers=True` in '{name}' to speed up the dataloader worker initialization.")
    elif dataloader.num_workers < 2 and upper_bound > 1:
        rank_zero_warn(f"The '{name}' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers={upper_bound}` in the `DataLoader` to improve performance.", category=PossibleUserWarning)
    if dataloader.persistent_workers and dataloader.pin_memory and (trainer.reload_dataloaders_every_n_epochs > 0):
        rank_zero_warn('The combination of `DataLoader(`pin_memory=True`, `persistent_workers=True`) and `Trainer(reload_dataloaders_every_n_epochs > 0)` can lead to instability due to limitations in PyTorch (https://github.com/pytorch/pytorch/issues/91252). We recommend setting `pin_memory=False` in this case.', category=PossibleUserWarning)