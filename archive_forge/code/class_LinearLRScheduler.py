from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
class LinearLRScheduler(ParlAILRScheduler):
    """
    Scheduler that decays linearly.
    """

    def __init__(self, optimizer, hard_reset, patience, decay, warmup_updates, warmup_rate, max_lr_steps):
        """
        max_lr_steps determines the cycle length of the linear annealing.

        It indicates the number of steps from 1.0 multiplier to 0.0
        """
        super().__init__(hard_reset, warmup_updates, warmup_rate)
        if max_lr_steps <= 0:
            raise ValueError('--lr-scheduler linear requires setting --max-lr-steps')
        self.max_lr_steps = max_lr_steps
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._linear_lr)

    def _linear_lr(self, step):
        lr_mult = max(0.0, 1.0 - step / self.max_lr_steps)
        return lr_mult

    def train_step(self, scheduler_steps):
        if scheduler_steps >= self.max_lr_steps:
            raise StopTrainException('End of Linear LR Schedule')
        self.scheduler.step(epoch=scheduler_steps)

    def valid_step(self, metrics_dict):
        pass