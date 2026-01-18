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
def _call_lightning_datamodule_hook(trainer: 'pl.Trainer', hook_name: str, *args: Any, **kwargs: Any) -> Any:
    log.debug(f'{trainer.__class__.__name__}: calling lightning datamodule hook: {hook_name}')
    if trainer.datamodule is None:
        raise TypeError('No `LightningDataModule` is available to call hooks on.')
    fn = getattr(trainer.datamodule, hook_name)
    if callable(fn):
        with trainer.profiler.profile(f'[LightningDataModule]{trainer.datamodule.__class__.__name__}.{hook_name}'):
            return fn(*args, **kwargs)
    return None