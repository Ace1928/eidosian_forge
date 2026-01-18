import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_loop_structure_after_tbptt_removal(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Adjusts the loop structure since it changed when the support for truncated backpropagation was removed. The
    optimizer loop and the manual loop were previously children of the training batch loop. After its removal, they
    became the children of the training epoch loop.

    Version: 2.0.0
    Commit: 7807454
    PR: #16337, #16172

    """
    if 'loops' not in checkpoint:
        return checkpoint
    if 'fit_loop' not in checkpoint['loops']:
        return checkpoint
    fit_loop = checkpoint['loops']['fit_loop']
    old_key_new_key_mapping = {'epoch_loop.batch_loop.manual_loop.optim_step_progress': 'epoch_loop.manual_loop.optim_step_progress', 'epoch_loop.batch_loop.manual_loop.state_dict': 'epoch_loop.manual_loop.state_dict', 'epoch_loop.batch_loop.optimizer_loop.optim_progress': 'epoch_loop.optimizer_loop.optim_progress', 'epoch_loop.batch_loop.optimizer_loop.state_dict': 'epoch_loop.optimizer_loop.state_dict'}
    for old, new in list(old_key_new_key_mapping.items()):
        if old in fit_loop:
            fit_loop[new] = fit_loop[old]
            del fit_loop[old]
    if 'epoch_loop.batch_loop.state_dict' in fit_loop and fit_loop['epoch_loop.batch_loop.state_dict']:
        fit_loop['epoch_loop.state_dict']['old_batch_loop_state_dict'] = fit_loop['epoch_loop.batch_loop.state_dict']
    fit_loop.pop('epoch_loop.batch_loop.state_dict', None)
    return checkpoint