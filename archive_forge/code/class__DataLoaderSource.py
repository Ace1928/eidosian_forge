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
class _DataLoaderSource:
    """Stores the information where the dataloaders come from.

    The source can be

    1. from a ``*_dataloader()`` method on the :class:`~pytorch_lightning.core.LightningModule`,
    2. from a ``*_dataloader()`` method on the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`,
    3. a direct instance of a :class:`~torch.utils.data.DataLoader` or supported collections thereof.

    Arguments:
        instance: A LightningModule, LightningDataModule, or (a collection of) iterable(s).
        name: A name for this dataloader source. If the instance is a module, the name corresponds to the hook
            that returns the desired dataloader(s).

    """
    instance: Optional[Union[TRAIN_DATALOADERS, EVAL_DATALOADERS, 'pl.LightningModule', 'pl.LightningDataModule']]
    name: str

    def dataloader(self) -> Union[TRAIN_DATALOADERS, EVAL_DATALOADERS]:
        """Returns the dataloader from the source.

        If the source is a module, the method with the corresponding :attr:`name` gets called.

        """
        if isinstance(self.instance, pl.LightningModule):
            return call._call_lightning_module_hook(self.instance.trainer, self.name, pl_module=self.instance)
        if isinstance(self.instance, pl.LightningDataModule):
            assert self.instance.trainer is not None
            return call._call_lightning_datamodule_hook(self.instance.trainer, self.name)
        assert self.instance is not None
        return self.instance

    def is_defined(self) -> bool:
        """Returns whether the source dataloader can be retrieved or not.

        If the source is a module it checks that the method with given :attr:`name` is overridden.

        """
        return not self.is_module() or is_overridden(self.name, self.instance)

    def is_module(self) -> bool:
        """Returns whether the DataLoader source is a LightningModule or a LightningDataModule.

        It does not check whether ``*_dataloader`` methods are actually overridden.

        """
        return isinstance(self.instance, (pl.LightningModule, pl.LightningDataModule))