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
def _replace_module(self, parent, child_name, new_module, child):
    setattr(parent, child_name, new_module)
    if hasattr(child, 'base_layer'):
        child = child.base_layer
    if not hasattr(new_module, 'base_layer'):
        new_module.weight = child.weight
        if hasattr(child, 'bias'):
            new_module.bias = child.bias
    if getattr(child, 'state', None) is not None:
        if hasattr(new_module, 'base_layer'):
            new_module.base_layer.state = child.state
        else:
            new_module.state = child.state
        new_module.to(child.weight.device)
    for name, module in new_module.named_modules():
        if self.prefix in name or 'ranknum' in name:
            weight = child.qweight if hasattr(child, 'qweight') else child.weight
            module.to(weight.device)