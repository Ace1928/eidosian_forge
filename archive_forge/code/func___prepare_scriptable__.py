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
def __prepare_scriptable__(self) -> nn.Module:
    if self._v1_transform_cls is None:
        raise RuntimeError(f'Transform {type(self).__name__} cannot be JIT scripted. torchscript is only supported for backward compatibility with transforms which are already in torchvision.transforms. For torchscript support (on tensors only), you can use the functional API instead.')
    return self._v1_transform_cls(**self._extract_params_for_v1_transform())