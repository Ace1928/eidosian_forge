import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str]]:
    should_stop = False
    reason = None
    if self.check_finite and (not torch.isfinite(current)):
        should_stop = True
        reason = f'Monitored metric {self.monitor} = {current} is not finite. Previous best value was {self.best_score:.3f}. Signaling Trainer to stop.'
    elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
        should_stop = True
        reason = f'Stopping threshold reached: {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}. Signaling Trainer to stop.'
    elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
        should_stop = True
        reason = f'Divergence threshold reached: {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}. Signaling Trainer to stop.'
    elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
        should_stop = False
        reason = self._improvement_message(current)
        self.best_score = current
        self.wait_count = 0
    else:
        self.wait_count += 1
        if self.wait_count >= self.patience:
            should_stop = True
            reason = f'Monitored metric {self.monitor} did not improve in the last {self.wait_count} records. Best score: {self.best_score:.3f}. Signaling Trainer to stop.'
    return (should_stop, reason)