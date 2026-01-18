from __future__ import annotations
import contextlib
import functools
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union
import torch
import torch.utils.hooks as hooks
from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
from .hooks import AlignDevicesHook
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
from .utils.constants import FSDP_PYTORCH_VERSION
from .utils.modeling import get_state_dict_offloaded_model
from .utils.other import is_compiled_module
from torch.distributed.algorithms.join import Join
@contextmanager
def accumulate(self, *models):
    """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            *models (list of `torch.nn.Module`):
                PyTorch Modules that were prepared with `Accelerator.prepare`. Models passed to `accumulate()` will
                skip gradient syncing during backward pass in distributed training

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=1)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, output in dataloader:
        ...     with accelerator.accumulate(model):
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
        ...         optimizer.zero_grad()
        ```
        """
    self._do_sync(force=self.gradient_state.plugin_kwargs.get('sync_each_batch', False))
    with contextlib.ExitStack() as cm_stack:
        for m in models:
            cm_stack.enter_context(contextlib.nullcontext() if self.sync_gradients else self.no_sync(m))
        yield