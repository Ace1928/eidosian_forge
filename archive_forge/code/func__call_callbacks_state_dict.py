import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Type, Union
from packaging.version import Version
import pytorch_lightning as pl
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from pytorch_lightning.callbacks import Checkpoint, EarlyStopping
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _call_callbacks_state_dict(trainer: 'pl.Trainer') -> Dict[str, dict]:
    """Called when saving a model checkpoint, calls and returns every callback's `state_dict`, keyed by
    `Callback.state_key`."""
    callback_state_dicts = {}
    for callback in trainer.callbacks:
        state_dict = callback.state_dict()
        if state_dict:
            callback_state_dicts[callback.state_key] = state_dict
    return callback_state_dicts