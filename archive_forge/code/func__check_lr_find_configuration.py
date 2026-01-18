from typing import TYPE_CHECKING, Literal, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
def _check_lr_find_configuration(trainer: 'pl.Trainer') -> None:
    from pytorch_lightning.callbacks.lr_finder import LearningRateFinder
    configured_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, LearningRateFinder)]
    if configured_callbacks:
        raise ValueError('Trainer is already configured with a `LearningRateFinder` callback.Please remove it if you want to use the Tuner.')