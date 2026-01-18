from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def _propogate_module_bias(module: nn.Module, mask: Tensor) -> Optional[Tensor]:
    """
    In the case that we need to propagate biases, this function will return the biases we need
    """
    if module.bias is not None:
        module.bias = nn.Parameter(cast(Tensor, module.bias)[mask])
    elif getattr(module, '_bias', None) is not None:
        module.bias = nn.Parameter(cast(Tensor, module._bias)[mask])
    if getattr(module, '_bias', None) is not None:
        pruned_biases = cast(Tensor, module._bias)[~mask]
    else:
        pruned_biases = None
    if hasattr(module, '_bias'):
        delattr(module, '_bias')
    return pruned_biases