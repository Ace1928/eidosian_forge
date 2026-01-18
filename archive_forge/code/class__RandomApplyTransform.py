from __future__ import annotations
import enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import PIL.Image
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.transforms.v2._utils import check_type, has_any, is_pure_tensor
from torchvision.utils import _log_api_usage_once
from .functional._utils import _get_kernel
class _RandomApplyTransform(Transform):

    def __init__(self, p: float=0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError('`p` should be a floating point value in the interval [0.0, 1.0].')
        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        self._check_inputs(flat_inputs)
        if torch.rand(1) >= self.p:
            return inputs
        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params([inpt for inpt, needs_transform in zip(flat_inputs, needs_transform_list) if needs_transform])
        flat_outputs = [self._transform(inpt, params) if needs_transform else inpt for inpt, needs_transform in zip(flat_inputs, needs_transform_list)]
        return tree_unflatten(flat_outputs, spec)