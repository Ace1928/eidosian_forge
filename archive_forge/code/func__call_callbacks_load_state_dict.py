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
def _call_callbacks_load_state_dict(trainer: 'pl.Trainer', checkpoint: Dict[str, Any]) -> None:
    """Called when loading a model checkpoint, calls every callback's `load_state_dict`."""
    callback_states: Optional[Dict[Union[Type, str], Dict]] = checkpoint.get('callbacks')
    if callback_states is None:
        return
    for callback in trainer.callbacks:
        state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
        if state:
            state = deepcopy(state)
            callback.load_state_dict(state)