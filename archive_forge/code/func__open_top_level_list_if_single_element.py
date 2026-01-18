from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
def _open_top_level_list_if_single_element(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    if spec.type == list and len(spec.children_specs) == 1:
        return spec.children_specs[0]
    return spec