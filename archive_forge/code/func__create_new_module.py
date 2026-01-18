from __future__ import annotations
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
@staticmethod
def _create_new_module(lora_config, adapter_name, target, **kwargs):
    dispatchers = []
    if is_bnb_available():
        from .bnb import dispatch_bnb_8bit
        dispatchers.append(dispatch_bnb_8bit)
    if is_bnb_4bit_available():
        from .bnb import dispatch_bnb_4bit
        dispatchers.append(dispatch_bnb_4bit)
    dispatchers.extend([dispatch_aqlm, dispatch_awq, dispatch_gptq, dispatch_megatron, dispatch_default])
    new_module = None
    for dispatcher in dispatchers:
        new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
        if new_module is not None:
            break
    if new_module is None:
        raise ValueError(f'Target module {target} is not supported. Currently, only the following modules are supported: `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`.')
    return new_module