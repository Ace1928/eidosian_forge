import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_loop_batches_that_stepped(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Sets the `_batches_that_stepped` default value for checkpoints before v1.6.5 which don't have this key.

    Version: 1.6.5
    Commit: c67b075
    PR: #13645

    """
    global_step = checkpoint['global_step']
    checkpoint['loops']['fit_loop']['epoch_loop.state_dict'].setdefault('_batches_that_stepped', global_step)
    return checkpoint