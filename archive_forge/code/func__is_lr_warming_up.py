from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
def _is_lr_warming_up(self):
    """
        Check if we're warming up the learning rate.
        """
    return hasattr(self, 'warmup_scheduler') and self.warmup_scheduler is not None and (self._number_training_updates <= self.warmup_updates)