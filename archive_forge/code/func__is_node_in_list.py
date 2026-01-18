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
def _is_node_in_list(node, modules, func_list, method_list, module_type_list):
    is_call_function = node.op == 'call_function' and node.target in func_list
    is_call_method = node.op == 'call_method' and node.target in method_list
    is_call_module = node.op == 'call_module' and type(modules[str(node.target)]) in module_type_list
    return (is_call_function, is_call_method, is_call_module)