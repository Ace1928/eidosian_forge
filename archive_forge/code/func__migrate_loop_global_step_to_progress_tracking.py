import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_loop_global_step_to_progress_tracking(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Sets the `global_step` value for checkpoints before v1.6 without the progress tracking state. It will be
    overwritten by the loop's state if it was also saved.

    Version: 1.6.0
    Commit: c67b075
    PR: #13645, #11805

    """
    global_step = checkpoint['global_step']
    checkpoint.setdefault('loops', {'fit_loop': _get_fit_loop_initial_state_1_6_0()})
    checkpoint['loops'].setdefault('fit_loop', _get_fit_loop_initial_state_1_6_0())
    optim_progress = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.optimizer_loop.optim_progress']
    optim_progress['optimizer']['step']['total']['completed'] = global_step
    optim_step_progress = checkpoint['loops']['fit_loop']['epoch_loop.batch_loop.manual_loop.optim_step_progress']
    optim_step_progress['total']['completed'] = global_step
    return checkpoint