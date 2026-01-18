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
def _call_lightning_module_hook(trainer: 'pl.Trainer', hook_name: str, *args: Any, pl_module: Optional['pl.LightningModule']=None, **kwargs: Any) -> Any:
    log.debug(f'{trainer.__class__.__name__}: calling lightning module hook: {hook_name}')
    pl_module = pl_module or trainer.lightning_module
    if pl_module is None:
        raise TypeError('No `LightningModule` is available to call hooks on.')
    fn = getattr(pl_module, hook_name)
    if not callable(fn):
        return None
    prev_fx_name = pl_module._current_fx_name
    pl_module._current_fx_name = hook_name
    with trainer.profiler.profile(f'[LightningModule]{pl_module.__class__.__name__}.{hook_name}'):
        output = fn(*args, **kwargs)
    pl_module._current_fx_name = prev_fx_name
    return output