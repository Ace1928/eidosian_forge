from abc import abstractmethod
from torch import optim
import numpy as np
from parlai.core.exceptions import StopTrainException
from parlai.utils.misc import warn_once
def _init_warmup_scheduler(self, optimizer, states):
    updates_so_far = states.get('number_training_updates', 0)
    if self.warmup_updates > 0 and (updates_so_far < self.warmup_updates or self.hard_reset):
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._warmup_lr)
    else:
        self.warmup_scheduler = None