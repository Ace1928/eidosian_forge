import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_loop_structure_after_dataloader_loop_removal(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The dataloader loops (``_DataLoaderLoop``, ``_PredictionLoop`, and ``_EvaluationLoop``) were flattened into the
    ``_EvaluationEpochLoop`` (now ``_EvaluationLoop``) and ``_PredictionEpochLoop`` (now ``_PredictionLoop``).

    Version: 2.0.0
    Commit: ec4f592ecfe238edd83185f6c6905fb1e2406d61
    PR: #16726

    """
    if 'loops' not in checkpoint:
        return checkpoint
    loops = checkpoint['loops']
    for loop_key in ('predict_loop', 'validate_loop', 'test_loop'):
        if loop_key not in loops:
            continue
        loop = loops[loop_key]
        loop.pop('dataloader_progress', None)
        epoch_loop_key = 'epoch_loop.'
        epoch_loop_dict = {k[len(epoch_loop_key):]: loop.pop(k) for k in list(loop) if k.startswith(epoch_loop_key)}
        loop.update(epoch_loop_dict)
    return checkpoint