from typing import Optional, Union
import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.profilers import (
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable, _habana_available_and_importable
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _init_debugging_flags(trainer: 'pl.Trainer', limit_train_batches: Optional[Union[int, float]], limit_val_batches: Optional[Union[int, float]], limit_test_batches: Optional[Union[int, float]], limit_predict_batches: Optional[Union[int, float]], fast_dev_run: Union[int, bool], overfit_batches: Union[int, float], val_check_interval: Optional[Union[int, float]], num_sanity_val_steps: int) -> None:
    if isinstance(fast_dev_run, int) and fast_dev_run < 0:
        raise MisconfigurationException(f'fast_dev_run={fast_dev_run!r} is not a valid configuration. It should be >= 0.')
    trainer.fast_dev_run = fast_dev_run
    if fast_dev_run == 1:
        trainer.fast_dev_run = True
    trainer.overfit_batches = _determine_batch_limits(overfit_batches, 'overfit_batches')
    overfit_batches_enabled = overfit_batches > 0
    if fast_dev_run:
        num_batches = int(fast_dev_run)
        if not overfit_batches_enabled:
            trainer.limit_train_batches = num_batches
            trainer.limit_val_batches = num_batches
        trainer.limit_test_batches = num_batches
        trainer.limit_predict_batches = num_batches
        trainer.fit_loop.epoch_loop.max_steps = num_batches
        trainer.num_sanity_val_steps = 0
        trainer.fit_loop.max_epochs = 1
        trainer.val_check_interval = 1.0
        trainer.check_val_every_n_epoch = 1
        trainer.loggers = [DummyLogger()] if trainer.loggers else []
        rank_zero_info(f'Running in `fast_dev_run` mode: will run the requested loop using {num_batches} batch(es). Logging and checkpointing is suppressed.')
    else:
        if not overfit_batches_enabled:
            trainer.limit_train_batches = _determine_batch_limits(limit_train_batches, 'limit_train_batches')
            trainer.limit_val_batches = _determine_batch_limits(limit_val_batches, 'limit_val_batches')
        trainer.limit_test_batches = _determine_batch_limits(limit_test_batches, 'limit_test_batches')
        trainer.limit_predict_batches = _determine_batch_limits(limit_predict_batches, 'limit_predict_batches')
        trainer.num_sanity_val_steps = float('inf') if num_sanity_val_steps == -1 else num_sanity_val_steps
        trainer.val_check_interval = _determine_batch_limits(val_check_interval, 'val_check_interval')
    if overfit_batches_enabled:
        trainer.limit_train_batches = overfit_batches
        trainer.limit_val_batches = overfit_batches