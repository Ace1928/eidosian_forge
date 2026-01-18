from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
def _get_unique_module_name(module_names: Dict[str, int], module_name: str) -> str:
    module_names.setdefault(module_name, 0)
    module_names[module_name] += 1
    return f'{module_name}_{module_names[module_name]}'