import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
class DummyOptim:
    """
    Dummy optimizer presents model parameters or param groups, this is primarily used to follow conventional training
    loop when optimizer config is specified in the deepspeed config file.

    Args:
        lr (float):
            Learning rate.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay (float):
            Weight decay.
        **kwargs:
            Other arguments.
    """

    def __init__(self, params, lr=0.001, weight_decay=0, **kwargs):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs