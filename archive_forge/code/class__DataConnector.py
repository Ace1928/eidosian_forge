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
class _DataConnector:

    def __init__(self, trainer: 'pl.Trainer'):
        self.trainer = trainer
        self._datahook_selector: Optional[_DataHookSelector] = None

    def on_trainer_init(self, val_check_interval: Optional[Union[int, float]], reload_dataloaders_every_n_epochs: int, check_val_every_n_epoch: Optional[int]) -> None:
        self.trainer.datamodule = None
        if check_val_every_n_epoch is not None and (not isinstance(check_val_every_n_epoch, int)):
            raise MisconfigurationException(f'`check_val_every_n_epoch` should be an integer, found {check_val_every_n_epoch!r}.')
        if check_val_every_n_epoch is None and isinstance(val_check_interval, float):
            raise MisconfigurationException(f'`val_check_interval` should be an integer when `check_val_every_n_epoch=None`, found {val_check_interval!r}.')
        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch
        if not isinstance(reload_dataloaders_every_n_epochs, int) or reload_dataloaders_every_n_epochs < 0:
            raise MisconfigurationException(f'`reload_dataloaders_every_n_epochs` should be an int >= 0, got {reload_dataloaders_every_n_epochs}.')
        self.trainer.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs

    def prepare_data(self) -> None:
        trainer = self.trainer
        local_rank_zero = trainer.local_rank == 0
        global_rank_zero = trainer.local_rank == 0 and trainer.node_rank == 0
        datamodule = trainer.datamodule
        lightning_module = trainer.lightning_module
        if datamodule is not None:
            dm_prepare_data_per_node = datamodule.prepare_data_per_node
            if dm_prepare_data_per_node and local_rank_zero or (not dm_prepare_data_per_node and global_rank_zero):
                call._call_lightning_datamodule_hook(trainer, 'prepare_data')
        if lightning_module is not None:
            lm_prepare_data_per_node = lightning_module.prepare_data_per_node
            if lm_prepare_data_per_node and local_rank_zero or (not lm_prepare_data_per_node and global_rank_zero):
                call._call_lightning_module_hook(trainer, 'prepare_data')

    def attach_data(self, model: 'pl.LightningModule', train_dataloaders: Optional[TRAIN_DATALOADERS]=None, val_dataloaders: Optional[EVAL_DATALOADERS]=None, test_dataloaders: Optional[EVAL_DATALOADERS]=None, predict_dataloaders: Optional[EVAL_DATALOADERS]=None, datamodule: Optional['pl.LightningDataModule']=None) -> None:
        self.attach_dataloaders(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, test_dataloaders=test_dataloaders, predict_dataloaders=predict_dataloaders)
        self.attach_datamodule(model, datamodule=datamodule)
        model.trainer = self.trainer

    def attach_dataloaders(self, model: 'pl.LightningModule', train_dataloaders: Optional[TRAIN_DATALOADERS]=None, val_dataloaders: Optional[EVAL_DATALOADERS]=None, test_dataloaders: Optional[EVAL_DATALOADERS]=None, predict_dataloaders: Optional[EVAL_DATALOADERS]=None) -> None:
        trainer = self.trainer
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.epoch_loop.val_loop._combined_loader = None
        trainer.validate_loop._combined_loader = None
        trainer.test_loop._combined_loader = None
        trainer.predict_loop._combined_loader = None
        trainer.fit_loop._data_source.instance = train_dataloaders if train_dataloaders is not None else model
        trainer.fit_loop.epoch_loop.val_loop._data_source.instance = val_dataloaders if val_dataloaders is not None else model
        trainer.validate_loop._data_source.instance = val_dataloaders if val_dataloaders is not None else model
        trainer.test_loop._data_source.instance = test_dataloaders if test_dataloaders is not None else model
        trainer.predict_loop._data_source.instance = predict_dataloaders if predict_dataloaders is not None else model

    def attach_datamodule(self, model: 'pl.LightningModule', datamodule: Optional['pl.LightningDataModule']=None) -> None:
        self._datahook_selector = _DataHookSelector(model, datamodule)
        if datamodule is None:
            return
        trainer = self.trainer
        trainer.fit_loop._data_source.instance = datamodule
        trainer.fit_loop.epoch_loop.val_loop._data_source.instance = datamodule
        trainer.validate_loop._data_source.instance = datamodule
        trainer.test_loop._data_source.instance = datamodule
        trainer.predict_loop._data_source.instance = datamodule
        trainer.datamodule = datamodule
        datamodule.trainer = trainer

    def _requires_distributed_sampler(self, dataloader: DataLoader) -> bool:
        if _graphcore_available_and_importable():
            from lightning_graphcore import IPUAccelerator
            is_ipu = isinstance(self.trainer.accelerator, IPUAccelerator)
        else:
            is_ipu = False
        return self.trainer._accelerator_connector.use_distributed_sampler and self.trainer._accelerator_connector.is_distributed and (not isinstance(dataloader.sampler, DistributedSampler)) and (not has_iterable_dataset(dataloader)) and (not is_ipu)

    def _prepare_dataloader(self, dataloader: object, shuffle: bool, mode: RunningStage) -> object:
        """This function handles the following functionalities:

        - Injecting a `DistributedDataSamplerWrapper` into the `DataLoader` if on a distributed environment
        - Wrapping the dataloader based on strategy-specific logic

        """
        if not isinstance(dataloader, DataLoader):
            return dataloader
        if _graphcore_available_and_importable():
            from lightning_graphcore import IPUAccelerator
            is_ipu = isinstance(self.trainer.accelerator, IPUAccelerator)
        else:
            is_ipu = False
        if self._requires_distributed_sampler(dataloader) or mode == RunningStage.PREDICTING or is_ipu:
            sampler = self._resolve_sampler(dataloader, shuffle=shuffle, mode=mode)
            return _update_dataloader(dataloader, sampler, mode=mode)
        return dataloader

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