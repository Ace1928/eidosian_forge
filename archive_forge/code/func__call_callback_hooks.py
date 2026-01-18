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
def _call_callback_hooks(trainer: 'pl.Trainer', hook_name: str, *args: Any, monitoring_callbacks: Optional[bool]=None, **kwargs: Any) -> None:
    log.debug(f'{trainer.__class__.__name__}: calling callback hook: {hook_name}')
    pl_module = trainer.lightning_module
    if pl_module:
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = hook_name
    callbacks = trainer.callbacks
    if monitoring_callbacks is True:
        callbacks = [cb for cb in callbacks if isinstance(cb, (EarlyStopping, Checkpoint))]
    elif monitoring_callbacks is False:
        callbacks = [cb for cb in callbacks if not isinstance(cb, (EarlyStopping, Checkpoint))]
    for callback in callbacks:
        fn = getattr(callback, hook_name)
        if callable(fn):
            with trainer.profiler.profile(f'[Callback]{callback.state_key}.{hook_name}'):
                fn(trainer, trainer.lightning_module, *args, **kwargs)
    if pl_module:
        pl_module._current_fx_name = prev_fx_name