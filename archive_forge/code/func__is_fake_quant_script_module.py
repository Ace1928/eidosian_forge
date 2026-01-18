import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
def _is_fake_quant_script_module(mod):
    """Return true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        suffix = mod._c.qualified_name.split('.', 1)[1]
        name = re.sub('\\.___torch_mangle_\\d+', '', suffix)
        return name == 'torch.ao.quantization.fake_quantize.FakeQuantize' or name == 'torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize'
    return False