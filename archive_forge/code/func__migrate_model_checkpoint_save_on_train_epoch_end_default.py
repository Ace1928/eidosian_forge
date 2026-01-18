import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_model_checkpoint_save_on_train_epoch_end_default(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The ``save_on_train_epoch_end`` was removed from the state-key of ``ModelCheckpoint`` in 1.9.0, and this
    migration drops it from the state-keys saved in the checkpoint dict so that the keys match when the Trainer loads
    the callback state.

    Version: 1.9.0
    Commit: f4ca56
    PR: #15300, #15606

    """
    if 'callbacks' not in checkpoint:
        return checkpoint

    def new_key(old_key: str) -> str:
        if not old_key.startswith('ModelCheckpoint'):
            return old_key
        return re.sub(", 'save_on_train_epoch_end': (None|True|False)", '', old_key)
    num_keys = len(checkpoint['callbacks'])
    new_callback_states = {new_key(old_key): state for old_key, state in checkpoint['callbacks'].items() if isinstance(old_key, str)}
    if len(new_callback_states) < num_keys:
        rank_zero_warn("You have multiple `ModelCheckpoint` callback states in this checkpoint, but we found state keys that would end up colliding with each other after an upgrade, which means we can't differentiate which of your checkpoint callbacks needs which states. At least one of your `ModelCheckpoint` callbacks will not be able to reload the state.", category=PossibleUserWarning)
        return checkpoint
    checkpoint['callbacks'] = new_callback_states
    return checkpoint