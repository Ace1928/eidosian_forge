import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
@staticmethod
def _create_from(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple['FunctionalModule', Tuple[Tensor, ...]]:
    model_copy = copy.deepcopy(model)
    params, param_names, names_map = extract_weights(model_copy)
    if disable_autograd_tracking:
        for param in params:
            param.requires_grad_(False)
    return (FunctionalModule(model_copy, param_names, names_map), params)