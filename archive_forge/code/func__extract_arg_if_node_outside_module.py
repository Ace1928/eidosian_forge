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
def _extract_arg_if_node_outside_module(arg: Any):
    if isinstance(arg, torch.fx.Node) and arg not in node_set:
        module_inputs[arg] = None