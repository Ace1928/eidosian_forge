from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
def _warmup_lr(self, step):
    """
        Return lr multiplier (on initial lr) for warmup scheduler.
        """
    start = self.warmup_rate
    end = 1.0
    progress = min(1.0, step / self.warmup_updates)
    lr_mult = start + (end - start) * progress
    return lr_mult