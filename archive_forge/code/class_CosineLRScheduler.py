from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
class CosineLRScheduler(ParlAILRScheduler):
    """
    Scheduler that decays by a cosine function.
    """

    def __init__(self, optimizer, hard_reset, patience, decay, warmup_updates, warmup_rate, max_lr_steps):
        """
        max_lr_steps determines the cycle length of the cosine annealing.

        It indicates the number of steps from 1.0 multiplier to 0.0, which corresponds
        to going from cos(0) to cos(pi)
        """
        super().__init__(hard_reset, warmup_updates, warmup_rate)
        if max_lr_steps <= 0:
            raise ValueError('--lr-scheduler cosine requires setting --max-lr-steps')
        self.max_lr_steps = max_lr_steps
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_lr_steps)

    def train_step(self, scheduler_steps):
        if scheduler_steps >= self.max_lr_steps:
            raise StopTrainException('End of Cosine LR Schedule')
        self.scheduler.step(epoch=scheduler_steps)

    def valid_step(self, metrics_dict):
        pass