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
def _lower_to_native_backend(model: GraphModule, qconfig_map: Dict[str, QConfigAny], node_name_to_scope: Dict[str, Tuple[str, type]]) -> GraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    _lower_static_weighted_ref_module(model, qconfig_map)
    _lower_static_weighted_ref_module_with_two_inputs(model, qconfig_map)
    _lower_dynamic_weighted_ref_module(model)
    _lower_weight_only_weighted_ref_module(model)
    _lower_static_weighted_ref_functional(model, qconfig_map)
    _lower_dynamic_weighted_ref_functional(model, qconfig_map)
    _lower_quantized_binary_op(model, qconfig_map)
    _lower_getattr_tensor_metadta_op(model)
    _lower_get_tensor_info_op(model)
    special_pattern_replacement(model)
    model.graph.eliminate_dead_code()
    model = fold_weight(model, node_name_to_scope)
    model.graph.eliminate_dead_code()
    model.recompile()
    model.graph.lint()
    return model