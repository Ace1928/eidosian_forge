import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_loop_structure_after_optimizer_loop_removal(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Adjusts the loop structure since it changed when the support for multiple optimizers in automatic optimization
    mode was removed. There is no longer a loop over optimizer, and hence no position to store for resuming the loop.

    Version: 2.0.0
    Commit: 6a56586
    PR: #16539, #16598

    """
    if 'loops' not in checkpoint:
        return checkpoint
    if 'fit_loop' not in checkpoint['loops']:
        return checkpoint
    fit_loop = checkpoint['loops']['fit_loop']
    if 'epoch_loop.optimizer_loop.optim_progress' in fit_loop:
        fit_loop['epoch_loop.optimizer_loop.optim_progress'].pop('optimizer_position', None)
    if 'epoch_loop.optimizer_loop.state_dict' in fit_loop:
        fit_loop['epoch_loop.automatic_optimization.state_dict'] = fit_loop.pop('epoch_loop.optimizer_loop.state_dict')
        fit_loop['epoch_loop.automatic_optimization.optim_progress'] = fit_loop.pop('epoch_loop.optimizer_loop.optim_progress')
    if 'epoch_loop.manual_loop.state_dict' in fit_loop:
        fit_loop['epoch_loop.manual_optimization.state_dict'] = fit_loop.pop('epoch_loop.manual_loop.state_dict')
        fit_loop['epoch_loop.manual_optimization.optim_step_progress'] = fit_loop.pop('epoch_loop.manual_loop.optim_step_progress')
    return checkpoint