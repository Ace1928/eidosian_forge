import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _scale_batch_size(trainer: 'pl.Trainer', mode: str='power', steps_per_trial: int=3, init_val: int=2, max_trials: int=25, batch_arg_name: str='batch_size') -> Optional[int]:
    """Iteratively try to find the largest batch size for a given model that does not give an out of memory (OOM)
    error.

    Args:
        trainer: A Trainer instance.
        mode: Search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practise a few are needed
        init_val: initial batch size to start the search with
        max_trials: max number of increases in batch size done before
           algorithm is terminated
        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

    """
    if trainer.fast_dev_run:
        rank_zero_warn('Skipping batch size scaler since `fast_dev_run` is enabled.')
        return None
    ckpt_path = os.path.join(trainer.default_root_dir, f'.scale_batch_size_{uuid.uuid4()}.ckpt')
    trainer.save_checkpoint(ckpt_path)
    params = __scale_batch_dump_params(trainer)
    __scale_batch_reset_params(trainer, steps_per_trial)
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()
    new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=init_val)
    if mode == 'power':
        new_size = _run_power_scaling(trainer, new_size, batch_arg_name, max_trials, params)
    elif mode == 'binsearch':
        new_size = _run_binary_scaling(trainer, new_size, batch_arg_name, max_trials, params)
    garbage_collection_cuda()
    log.info(f'Finished batch size finder, will continue with full run using batch size {new_size}')
    __scale_batch_restore_params(trainer, params)
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()
    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)
    return new_size