import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union
import pytorch_lightning as pl
from lightning_fabric.utilities.registry import _load_external_callbacks
from pytorch_lightning.callbacks import (
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder
from pytorch_lightning.callbacks.lr_finder import LearningRateFinder
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info
def _attach_model_callbacks(self) -> None:
    """Attaches the callbacks defined in the model.

        If a callback returned by the model's configure_callback method has the same type as one or several
        callbacks already present in the trainer callbacks list, it will replace them.
        In addition, all :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks
        will be pushed to the end of the list, ensuring they run last.

        """
    trainer = self.trainer
    model_callbacks = call._call_lightning_module_hook(trainer, 'configure_callbacks')
    if not model_callbacks:
        return
    model_callbacks = [model_callbacks] if not isinstance(model_callbacks, Sequence) else model_callbacks
    model_callback_types = {type(c) for c in model_callbacks}
    trainer_callback_types = {type(c) for c in trainer.callbacks}
    trainer_callback_types.discard(Callback)
    override_types = set()
    for model_cb in model_callback_types:
        for trainer_cb in trainer_callback_types:
            if issubclass(model_cb, trainer_cb):
                override_types.add(trainer_cb)
                break
    if override_types:
        rank_zero_info(f'The following callbacks returned in `LightningModule.configure_callbacks` will override existing callbacks passed to Trainer: {', '.join(sorted((t.__name__ for t in override_types)))}')
    all_callbacks = [c for c in trainer.callbacks if type(c) not in override_types]
    all_callbacks.extend(model_callbacks)
    all_callbacks = _CallbackConnector._reorder_callbacks(all_callbacks)
    trainer.callbacks = all_callbacks