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
def _resolve_overfit_batches(combined_loader: CombinedLoader, mode: RunningStage) -> None:
    all_have_sequential_sampler = all((isinstance(dl.sampler, SequentialSampler) for dl in combined_loader.flattened if hasattr(dl, 'sampler')))
    if all_have_sequential_sampler:
        return
    rank_zero_warn(f'You requested to overfit but enabled {mode.dataloader_prefix} dataloader shuffling. We are turning off the {mode.dataloader_prefix} dataloader shuffling for you.')
    updated = [_update_dataloader(dl, sampler=SequentialSampler(dl.dataset), mode=mode) if hasattr(dl, 'dataset') else dl for dl in combined_loader.flattened]
    combined_loader.flattened = updated