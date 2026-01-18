import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class GradientAccumulationPlugin(KwargsHandler):
    """
    A plugin to configure gradient accumulation behavior. You can only pass one of `gradient_accumulation_plugin` or
    `gradient_accumulation_steps` to [`Accelerator`]. Passing both raises an error.

    Parameters:
        num_steps (`int`):
            The number of steps to accumulate gradients for.
        adjust_scheduler (`bool`, *optional*, defaults to `True`):
            Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be
            `True` if the used scheduler was not adjusted for gradient accumulation.
        sync_with_dataloader (`bool`, *optional*, defaults to `True`):
            Whether to synchronize setting the gradients when at the end of the dataloader.
        sync_each_batch (`bool`, *optional*):
                Whether to synchronize setting the gradients at each data batch. Seting to `True` may reduce memory
                requirements when using gradient accumulation with distributed training, at expense of speed.

    Example:

    ```python
    from accelerate.utils import GradientAccumulationPlugin

    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2)
    accelerator = Accelerator(gradient_accumulation_plugin=gradient_accumulation_plugin)
    ```
    """
    num_steps: int = field(default=None, metadata={'help': 'The number of steps to accumulate gradients for.'})
    adjust_scheduler: bool = field(default=True, metadata={'help': 'Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be `True` if the used scheduler was not adjusted for gradient accumulation.'})
    sync_with_dataloader: bool = field(default=True, metadata={'help': "Whether to synchronize setting the gradients when at the end of the dataloader. Should only be set to `False` if you know what you're doing."})
    sync_each_batch: bool = field(default=False, metadata={'help': 'Whether to synchronize setting the gradients at each data batch. Setting to `True` may reduce memory requirements when using gradient accumulation with distributed training, at expense of speed.'})