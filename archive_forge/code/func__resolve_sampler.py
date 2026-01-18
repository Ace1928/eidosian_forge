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
def _resolve_sampler(self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage]=None) -> Union[Sampler, Iterable]:
    if self._requires_distributed_sampler(dataloader):
        distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
        assert distributed_sampler_kwargs is not None
        sampler = _get_distributed_sampler(dataloader, shuffle, mode=mode, overfit_batches=self.trainer.overfit_batches, **distributed_sampler_kwargs)
        trainer_fn = self.trainer.state.fn
        if isinstance(sampler, DistributedSampler) and sampler.num_replicas > 1 and (trainer_fn in (TrainerFn.VALIDATING, TrainerFn.TESTING)):
            rank_zero_warn(f'Using `DistributedSampler` with the dataloaders. During `trainer.{trainer_fn.value}()`, it is recommended to use `Trainer(devices=1, num_nodes=1)` to ensure each sample/batch gets evaluated exactly once. Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have same batch size in case of uneven inputs.', category=PossibleUserWarning)
        return sampler
    return dataloader.sampler