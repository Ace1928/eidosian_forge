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
def _process_dataloader(trainer: 'pl.Trainer', trainer_fn: TrainerFn, stage: RunningStage, dataloader: object) -> object:
    if stage != RunningStage.TRAINING:
        is_shuffled = _is_dataloader_shuffled(dataloader)
        if is_shuffled:
            rank_zero_warn(f"Your `{stage.dataloader_prefix}_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.", category=PossibleUserWarning)
    else:
        is_shuffled = True
    dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=is_shuffled, mode=stage)
    dataloader = trainer.strategy.process_dataloader(dataloader)
    _worker_check(trainer=trainer, dataloader=dataloader, name=f'{stage.dataloader_prefix}_dataloader')
    _auto_add_worker_init_fn(dataloader, trainer.global_rank)
    if trainer_fn != TrainerFn.FITTING:
        _set_sampler_epoch(dataloader, trainer.fit_loop.epoch_progress.current.processed)
    return dataloader