import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
def _run_early_stopping_check(self, trainer: 'pl.Trainer') -> None:
    """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
    logs = trainer.callback_metrics
    if trainer.fast_dev_run or not self._validate_condition_metric(logs):
        return
    current = logs[self.monitor].squeeze()
    should_stop, reason = self._evaluate_stopping_criteria(current)
    should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
    trainer.should_stop = trainer.should_stop or should_stop
    if should_stop:
        self.stopped_epoch = trainer.current_epoch
    if reason and self.verbose:
        self._log_info(trainer, reason, self.log_rank_zero_only)