from typing import Any, List, Optional, Union
import torch
from torch import nn
def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)
    for key, value in reassign.items():
        module._modules[key] = value