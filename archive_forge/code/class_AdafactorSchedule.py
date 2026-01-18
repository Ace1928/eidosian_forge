import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version
class AdafactorSchedule(LambdaLR):
    """
    Since [`~optimization.Adafactor`] performs its own scheduling, if the training loop relies on a scheduler (e.g.,
    for logging), this class creates a proxy object that retrieves the current lr values from the optimizer.

    It returns `initial_lr` during startup and the actual `lr` during stepping.
    """

    def __init__(self, optimizer, initial_lr=0.0):

        def lr_lambda(_):
            return initial_lr
        for group in optimizer.param_groups:
            group['initial_lr'] = initial_lr
        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group['initial_lr']

    def get_lr(self):
        opt = self.optimizer
        lrs = [opt._get_lr(group, opt.state[group['params'][0]]) for group in opt.param_groups if group['params'][0].grad is not None]
        if len(lrs) == 0:
            lrs = self.base_lrs
        return lrs