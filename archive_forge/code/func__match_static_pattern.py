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
def _match_static_pattern(node: Node, modules: Dict[str, nn.Module], qconfig_map: Dict[str, QConfigAny], matching_modules_or_ops: List[Callable], dequantize_node_arg_indices: List[int]) -> Union[Tuple[Node, Node, Node], Tuple[None, None, None]]:
    """
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 3-tuple of:
      1) q_node: the quantize node,
      2) relu_node: a relu node wrapping the ref_node, and
      3) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 3-tuple of (None, None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
      dequantize_node_arg_indices: A list of indices in the reference node args where dequantize
          nodes may be present. An empty list means skipping the check for dequantize nodes.
    """
    SKIP_LOWERING_VALUE = (None, None, None)
    if node.op != 'call_function' or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    assert isinstance(ref_node, Node)
    if ref_node.op == 'call_function' and ref_node.target in (F.relu, torch.relu) or (ref_node.op == 'call_module' and type(_get_module(ref_node, modules)) == nn.ReLU):
        relu_node = ref_node
        ref_node = relu_node.args[0]
        assert isinstance(ref_node, Node)
    else:
        relu_node = None
    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE
    if isinstance(matching_modules_or_ops[0], type) and issubclass(matching_modules_or_ops[0], nn.Module):
        expected_op = 'call_module'
        match_key = type(_get_module(ref_node, modules))
    else:
        expected_op = 'call_function'
        match_key = ref_node.target
    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE
    matched_dequantize = False
    for i in dequantize_node_arg_indices:
        assert i < len(ref_node.args), f"Dequantize index {i} exceeded reference node's arg length {len(ref_node.args)}"
        arg = ref_node.args[i]
        if is_dequantize_node(arg):
            matched_dequantize = True
        elif isinstance(arg, Node):
            return SKIP_LOWERING_VALUE
    if not matched_dequantize:
        return SKIP_LOWERING_VALUE
    return (q_node, relu_node, ref_node)