import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_model_checkpoint_early_stopping(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The checkpoint and early stopping keys were renamed.

    Version: 0.10.0
    Commit: a5d1176

    """
    keys_mapping = {'checkpoint_callback_best_model_score': (ModelCheckpoint, 'best_model_score'), 'checkpoint_callback_best_model_path': (ModelCheckpoint, 'best_model_path'), 'checkpoint_callback_best': (ModelCheckpoint, 'best_model_score'), 'early_stop_callback_wait': (EarlyStopping, 'wait_count'), 'early_stop_callback_patience': (EarlyStopping, 'patience')}
    checkpoint['callbacks'] = checkpoint.get('callbacks') or {}
    for key, new_path in keys_mapping.items():
        if key in checkpoint:
            value = checkpoint[key]
            callback_type, callback_key = new_path
            checkpoint['callbacks'][callback_type] = checkpoint['callbacks'].get(callback_type) or {}
            checkpoint['callbacks'][callback_type][callback_key] = value
            del checkpoint[key]
    return checkpoint