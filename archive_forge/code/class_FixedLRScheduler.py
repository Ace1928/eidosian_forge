from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
class FixedLRScheduler(ParlAILRScheduler):
    """
    Scheduler that decays by a fixed multiplicative rate at each valid step.
    """

    def __init__(self, optimizer, hard_reset, patience, decay, warmup_updates, warmup_rate):
        super().__init__(hard_reset, warmup_updates, warmup_rate)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, patience, gamma=decay)

    def train_step(self, scheduler_steps):
        pass

    def valid_step(self, metrics_dict):
        if self._is_lr_warming_up():
            return
        self.scheduler.step()