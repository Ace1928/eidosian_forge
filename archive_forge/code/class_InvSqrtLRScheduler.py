from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
class InvSqrtLRScheduler(ParlAILRScheduler):
    """
    Scheduler that decays at an inverse square root rate.
    """

    def __init__(self, optimizer, hard_reset, patience, decay, warmup_updates, warmup_rate, invsqrt_lr_decay_gamma):
        """
        invsqrt_lr_decay_gamma determines the cycle length of the inverse square root
        scheduler.

        When steps taken == invsqrt_lr_decay_gamma, the lr multiplier is 1
        """
        super().__init__(hard_reset, warmup_updates, warmup_rate)
        self.invsqrt_lr_decay_gamma = invsqrt_lr_decay_gamma
        if invsqrt_lr_decay_gamma <= 0:
            warn_once('--lr-scheduler invsqrt requires a value for --invsqrt-lr-decay-gamma. Defaulting to set gamma to --warmup-updates value for backwards compatibility.')
            self.invsqrt_lr_decay_gamma = self.warmup_updates
        self.decay_factor = np.sqrt(max(1, self.invsqrt_lr_decay_gamma))
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._invsqrt_lr)

    def _invsqrt_lr(self, step):
        return self.decay_factor / np.sqrt(max(1, self.invsqrt_lr_decay_gamma + step))

    def train_step(self, scheduler_steps):
        self.scheduler.step(epoch=scheduler_steps)

    def valid_step(self, metrics_dict):
        pass