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
def _call_callbacks_on_save_checkpoint(trainer: 'pl.Trainer', checkpoint: Dict[str, Any]) -> None:
    """Called when saving a model checkpoint, calls every callback's `on_save_checkpoint` hook."""
    pl_module = trainer.lightning_module
    if pl_module:
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = 'on_save_checkpoint'
    for callback in trainer.callbacks:
        with trainer.profiler.profile(f'[Callback]{callback.state_key}.on_save_checkpoint'):
            callback.on_save_checkpoint(trainer, trainer.lightning_module, checkpoint)
    if pl_module:
        pl_module._current_fx_name = prev_fx_name