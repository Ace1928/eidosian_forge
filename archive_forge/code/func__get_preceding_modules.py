from logging import getLogger
from typing import Optional, Union
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D
from .constants import BLOCK_PATTERNS, SEQLEN_KEYS_TRANFORMERS
def _get_preceding_modules(model: nn.Module, module_name: str, name: str=''):
    nonlocal stop_adding
    for name_bis, child in model.named_children():
        new_name = name + '.' + name_bis if name != '' else name_bis
        if new_name == module_name:
            stop_adding = True
            break
        _get_preceding_modules(child, module_name, name=new_name)
    if not stop_adding:
        previous_module_name.append(name)
    return previous_module_name