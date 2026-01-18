import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
class MegatronLMDummyScheduler:
    """
    Dummy scheduler presents model parameters or param groups, this is primarily used to follow conventional training
    loop when scheduler config is specified in the deepspeed config file.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        total_num_steps (int):
            Total number of steps.
        warmup_num_steps (int):
            Number of steps for warmup.
        **kwargs:
            Other arguments.
    """

    def __init__(self, optimizer, total_num_steps=None, warmup_num_steps=0, **kwargs):
        self.optimizer = optimizer
        self.total_num_steps = total_num_steps
        self.warmup_num_steps = warmup_num_steps
        self.kwargs = kwargs