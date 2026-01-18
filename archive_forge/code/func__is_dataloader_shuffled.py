import inspect
from dataclasses import fields
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Sized, Tuple, Union
import torch
from lightning_utilities.core.apply_func import is_dataclass_instance
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler, Sampler, SequentialSampler
from typing_extensions import TypeGuard
import pytorch_lightning as pl
from lightning_fabric.utilities.data import (
from pytorch_lightning.overrides.distributed import _IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
def _is_dataloader_shuffled(dataloader: object) -> bool:
    if hasattr(dataloader, '__pl_saved_kwargs'):
        if 'shuffle' in dataloader.__pl_saved_kwargs:
            return dataloader.__pl_saved_kwargs['shuffle']
        if 'shuffle' in dataloader.__pl_saved_arg_names:
            return dataloader.__pl_saved_args[dataloader.__pl_saved_arg_names.index('shuffle')]
    if hasattr(dataloader, 'dataset') and isinstance(dataloader.dataset, IterableDataset):
        return False
    if not hasattr(dataloader, 'sampler'):
        return False
    sampler = dataloader.sampler
    if isinstance(sampler, SequentialSampler):
        return False
    return isinstance(sampler, RandomSampler)