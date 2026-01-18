import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _maybe_report_no_init(module, name):
    if len(list(module.named_children())) == 0 and (hasattr(module, 'weight') or hasattr(module, 'bias')):
        if isinstance(module, torch.nn.LayerNorm):
            return
        if isinstance(module, torch.nn.Embedding):
            return
        logger.warning(f'Not initializing weights in {name}, this could be a mistake.\nModule {module}')
        if _assert_if_not_initialized:
            assert False, f'Uninitialized weight found in {module}.' + ' If you have a custom module, please provide a `init_weights()` method'