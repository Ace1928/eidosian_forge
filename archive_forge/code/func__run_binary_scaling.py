import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _run_binary_scaling(trainer: 'pl.Trainer', new_size: int, batch_arg_name: str, max_trials: int, params: Dict[str, Any]) -> int:
    """Batch scaling mode where the size is initially is doubled at each iteration until an OOM error is encountered.

    Hereafter, the batch size is further refined using a binary search

    """
    low = 1
    high = None
    count = 0
    while True:
        garbage_collection_cuda()
        _reset_progress(trainer)
        try:
            _try_loop_run(trainer, params)
            count += 1
            if count > max_trials:
                break
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc='succeeded')
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc='succeeded')
            if not changed:
                break
            _reset_dataloaders(trainer)
        except RuntimeError as exception:
            if is_oom_error(exception):
                garbage_collection_cuda()
                high = new_size
                midval = (high + low) // 2
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc='failed')
                _reset_dataloaders(trainer)
                if high - low <= 1:
                    break
            else:
                raise
    return new_size