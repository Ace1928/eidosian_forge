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
@register_annotator('add_relu')
def _annotate_add_relu(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    fused_partitions = find_sequential_partitions(gm, [torch.add, torch.nn.ReLU], filter_fn)
    annotated_partitions = []
    for fused_partition in fused_partitions:
        add_partition, relu_partition = fused_partition
        annotated_partitions.append(add_partition.nodes + relu_partition.nodes)
        if len(relu_partition.output_nodes) > 1:
            raise ValueError('Relu partition has more than one output node')
        relu_node = relu_partition.output_nodes[0]
        if len(add_partition.output_nodes) > 1:
            raise ValueError('add partition has more than one output node')
        add_node = add_partition.output_nodes[0]
        if _is_annotated([relu_node, add_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            input_qspec_map[input_act0] = input_act_qspec
        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            input_qspec_map[input_act1] = input_act_qspec
        add_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        relu_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions