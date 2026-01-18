import os
import shutil
import sys
from collections import ChainMap, OrderedDict, defaultdict
from typing import Any, DefaultDict, Iterable, Iterator, List, Optional, Tuple, Union
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.utilities.data import _set_sampler_epoch
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.progress import _BatchProgress
from pytorch_lightning.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import SIGTERMException
from pytorch_lightning.utilities.model_helpers import _ModuleMode, is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def _evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int, dataloader_iter: Optional[Iterator]) -> None:
    """Runs the actual evaluation step together with all the necessary bookkeeping and the hooks tied to it.

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch.
            dataloader_idx: the index of the dataloader producing the current batch.
            dataloader_iter: The iterator if using this step flavor.

        """
    trainer = self.trainer
    data_fetcher = self._data_fetcher
    assert data_fetcher is not None
    if not (using_dataloader_iter := isinstance(data_fetcher, _DataLoaderIterDataFetcher)):
        batch = trainer.precision_plugin.convert_input(batch)
        batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = call._call_strategy_hook(trainer, 'batch_to_device', batch, dataloader_idx=dataloader_idx)
    hook_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self._is_sequential and self.num_dataloaders > 1 else None)
    self.batch_progress.increment_ready()
    trainer._logger_connector.on_batch_start(batch, dataloader_idx if self._is_sequential and self.num_dataloaders > 1 else None)
    hook_name = 'on_test_batch_start' if trainer.testing else 'on_validation_batch_start'
    call._call_callback_hooks(trainer, hook_name, *hook_kwargs.values())
    call._call_lightning_module_hook(trainer, hook_name, *hook_kwargs.values())
    self.batch_progress.increment_started()
    hook_name = 'test_step' if trainer.testing else 'validation_step'
    step_args = self._build_step_args_from_hook_kwargs(hook_kwargs, hook_name) if not using_dataloader_iter else (dataloader_iter,)
    output = call._call_strategy_hook(trainer, hook_name, *step_args)
    self.batch_progress.increment_processed()
    if using_dataloader_iter:
        batch = data_fetcher._batch
        batch_idx = data_fetcher._batch_idx
        dataloader_idx = data_fetcher._dataloader_idx
        hook_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self._is_sequential and self.num_dataloaders > 1 else None)
    hook_name = 'on_test_batch_end' if trainer.testing else 'on_validation_batch_end'
    call._call_callback_hooks(trainer, hook_name, output, *hook_kwargs.values())
    call._call_lightning_module_hook(trainer, hook_name, output, *hook_kwargs.values())
    trainer._logger_connector.on_batch_end()
    self.batch_progress.increment_completed()
    if not trainer.sanity_checking:
        self._has_run = True
        trainer._logger_connector.update_eval_step_metrics(self._seen_batches_per_dataloader[dataloader_idx])
        self._seen_batches_per_dataloader[dataloader_idx] += 1
    if not self.batch_progress.is_last_batch and trainer.received_sigterm:
        raise SIGTERMException