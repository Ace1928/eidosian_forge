import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization.quantizer.utils import (
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
@register_annotator('conv_relu')
def _annotate_conv_relu(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != 'call_function' or n.target not in [torch.ops.aten.relu.default, torch.ops.aten.relu_.default]:
            continue
        relu_node = n
        maybe_conv_node = n.args[0]
        if not isinstance(maybe_conv_node, Node) or maybe_conv_node.op != 'call_function' or maybe_conv_node.target not in [torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default]:
            continue
        conv_node = maybe_conv_node
        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)
        partition = [relu_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)
        if _is_annotated(partition):
            continue
        if filter_fn and any((not filter_fn(n) for n in partition)):
            continue
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        relu_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions