from collections import OrderedDict
from typing import Any, Iterator, List, Optional, Union
import torch
from lightning_utilities import WarningCache
import pytorch_lightning as pl
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.progress import _Progress
from pytorch_lightning.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from pytorch_lightning.overrides.distributed import _IndexBatchSamplerWrapper
from pytorch_lightning.strategies.launchers import _MultiProcessingLauncher
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT
def _store_data_for_prediction_writer(self, batch_idx: int, dataloader_idx: int) -> bool:
    prediction_writers = [cb for cb in self.trainer.callbacks if isinstance(cb, BasePredictionWriter)]
    any_on_epoch = any((cb.interval.on_epoch for cb in prediction_writers))
    any_on_batch = any((cb.interval.on_batch for cb in prediction_writers))
    if any_on_batch or any_on_epoch:
        combined_loader = self._combined_loader
        assert combined_loader is not None
        dataloader = combined_loader.flattened[dataloader_idx]
        batch_indices = self._get_batch_indices(dataloader)
        if not batch_indices:
            return any_on_epoch
        batch_indices = batch_indices[batch_idx]
        if any_on_epoch:
            self.epoch_batch_indices[dataloader_idx].append(batch_indices)
        if any_on_batch:
            self.current_batch_indices = batch_indices
    return any_on_epoch