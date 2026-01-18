from typing import TYPE_CHECKING, Literal, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
def _check_scale_batch_size_configuration(trainer: 'pl.Trainer') -> None:
    if trainer._accelerator_connector.is_distributed:
        raise ValueError('Tuning the batch size is currently not supported with distributed strategies.')
    from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder
    configured_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, BatchSizeFinder)]
    if configured_callbacks:
        raise ValueError('Trainer is already configured with a `BatchSizeFinder` callback.Please remove it if you want to use the Tuner.')