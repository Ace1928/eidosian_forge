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
def _call_callbacks_on_load_checkpoint(trainer: 'pl.Trainer', checkpoint: Dict[str, Any]) -> None:
    """Called when loading a model checkpoint.

    Calls every callback's `on_load_checkpoint` hook. We have a dedicated function for this rather than using
    `_call_callback_hooks` because we have special logic for getting callback_states.

    """
    pl_module = trainer.lightning_module
    if pl_module:
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = 'on_load_checkpoint'
    callback_states: Optional[Dict[Union[Type, str], Dict]] = checkpoint.get('callbacks')
    if callback_states is None:
        return
    is_legacy_ckpt = Version(checkpoint['pytorch-lightning_version']) < Version('1.5.0dev')
    current_callbacks_keys = {cb._legacy_state_key if is_legacy_ckpt else cb.state_key for cb in trainer.callbacks}
    difference = callback_states.keys() - current_callbacks_keys
    if difference:
        rank_zero_warn(f'Be aware that when using `ckpt_path`, callbacks used to create the checkpoint need to be provided during `Trainer` instantiation. Please add the following callbacks: {list(difference)}.')
    for callback in trainer.callbacks:
        with trainer.profiler.profile(f'[Callback]{callback.state_key}.on_load_checkpoint'):
            callback.on_load_checkpoint(trainer, trainer.lightning_module, checkpoint)
    if pl_module:
        pl_module._current_fx_name = prev_fx_name