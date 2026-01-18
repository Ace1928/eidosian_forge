import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _insert_quantize_per_tensor_node(prev_node_c: Node, node_a: Node, gm_b: GraphModule, graph_c: Graph, scale: Union[torch.Tensor, float], zero_point: Union[torch.Tensor, int], dtype_cast_name: str) -> Node:
    scale_node_name = get_new_attr_name_with_prefix(node_a.name + '_input_scale_')(gm_b)
    setattr(gm_b, scale_node_name, scale)
    scale_node = graph_c.create_node('get_attr', scale_node_name, (), {}, scale_node_name)
    zero_point_node_name = get_new_attr_name_with_prefix(node_a.name + '_input_zero_point_')(gm_b)
    setattr(gm_b, zero_point_node_name, zero_point)
    zero_point_node = graph_c.create_node('get_attr', zero_point_node_name, (), {}, zero_point_node_name)
    return graph_c.create_node('call_function', torch.quantize_per_tensor, (prev_node_c, scale_node, zero_point_node, torch.quint8), {}, dtype_cast_name)