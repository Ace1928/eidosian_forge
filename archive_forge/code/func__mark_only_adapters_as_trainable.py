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
def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if self.prefix not in n:
            p.requires_grad = False
    for active_adapter in self.active_adapters:
        bias = self.peft_config[active_adapter].bias
        if bias == 'none':
            continue
        if bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoraLayer) and hasattr(m, 'bias') and (m.bias is not None):
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError(f'Requested bias: {bias}, is not implemented.')