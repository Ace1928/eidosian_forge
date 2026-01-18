import torch
from torch.fx import map_arg, Node
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.fx import GraphModule
from .utils import (
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
import operator
def _load_packed_weight(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    attrs_to_pop = []
    for attr_name in state_dict:
        if attr_name.startswith('_packed_weight') and isinstance(state_dict[attr_name], torch._C.ScriptObject):
            setattr(self, attr_name, state_dict[attr_name])
            attrs_to_pop.append(attr_name)
    for attr_name in attrs_to_pop:
        state_dict.pop(attr_name)