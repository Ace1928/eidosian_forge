import warnings
from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict
import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.utils as utils

        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``
        and it can be divided by ``period``, where ``step`` is increased by 1
        at each iteration in the training loop.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.

        