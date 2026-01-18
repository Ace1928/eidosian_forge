from __future__ import annotations
import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
@contextmanager
def _enable_peft_forward_hooks(self, *args, **kwargs):
    adapter_names = kwargs.pop('adapter_names', None)
    if adapter_names is None:
        yield
        return
    if self.training:
        raise ValueError('Cannot pass `adapter_names` when the model is in training mode.')
    hook_handles = []
    for module in self.modules():
        if isinstance(module, LoraLayer):
            pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
            handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
            hook_handles.append(handle)
    yield
    for handle in hook_handles:
        handle.remove()