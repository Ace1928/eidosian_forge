from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
def _linear_lr(self, step):
    lr_mult = max(0.0, 1.0 - step / self.max_lr_steps)
    return lr_mult