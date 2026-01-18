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
def _prepare_msamp(self, *args):
    if not is_msamp_available():
        raise ImportError("MS-AMP was not found on your system. Please ensure that MS-AMP is available  or choose `'te'` as the backend for FP8 mixed precision training.")
    else:
        import msamp
    model, optimizer = (None, None)
    num_models, num_optimizers = (0, 0)
    result = [obj for obj in args]
    for obj in result:
        if isinstance(obj, torch.nn.Module):
            model = obj
            num_models += 1
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = obj
            num_optimizers += 1
    if optimizer is None or model is None:
        raise ValueError('You must pass a model and an optimizer together to `accelerate.prepare()` when using MS-AMP.')
    elif num_models > 1 or num_optimizers > 1:
        raise ValueError(f"You can't use multiple models ({num_models}) or optimizers {num_optimizers} with MS-AMP.")
    else:
        model, optimizer = msamp.initialize(model, optimizer, opt_level=self.fp8_recipe_handler.opt_level)
    for i in range(len(result)):
        if isinstance(result[i], torch.nn.Module):
            result[i] = model
        elif isinstance(result[i], torch.optim.Optimizer):
            result[i] = optimizer
    return tuple(result)