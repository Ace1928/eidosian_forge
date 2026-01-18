import enum
import operator
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from typing import Tuple, Callable, Dict, Set, List, Optional, Union
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process
from .ns_types import NSNodeTargetType, NSResultsType
def _get_scale_zp_from_function_args(node, gm, scale_arg_idx, zp_arg_idx):
    scale_node = get_normalized_nth_input(node, gm, scale_arg_idx)
    zp_node = get_normalized_nth_input(node, gm, zp_arg_idx)
    assert isinstance(scale_node, Node) and isinstance(scale_node.target, str)
    assert isinstance(zp_node, Node) and isinstance(zp_node.target, str)
    scale_obj = getattr_from_fqn(gm, scale_node.target)
    zp_obj = getattr_from_fqn(gm, zp_node.target)
    return (scale_obj, zp_obj)