from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def _remove_bias_handles(module: nn.Module) -> None:
    if hasattr(module, '_forward_hooks'):
        bias_hooks: List[int] = []
        for key, hook in module._forward_hooks.items():
            if isinstance(hook, BiasHook):
                bias_hooks.append(key)
        for key in bias_hooks:
            del module._forward_hooks[key]