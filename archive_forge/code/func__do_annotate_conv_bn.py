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
def _do_annotate_conv_bn(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]], has_relu: bool) -> List[List[Node]]:
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    def get_pattern(conv_fn: Callable, relu_is_inplace: bool):

        def _conv_bn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_rm, bn_rv):
            conv = conv_fn(x, conv_weight, conv_bias)
            bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            if has_relu:
                output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
            else:
                output = bn
            return (output, {'input': x, 'conv': conv, 'weight': conv_weight, 'bias': conv_bias, 'output': output})
        return _conv_bn
    gm.graph.eliminate_dead_code()
    gm.recompile()
    matches = []
    combinations = [(F.conv1d, _conv1d_bn_example_inputs), (F.conv2d, _conv2d_bn_example_inputs)]
    combinations = itertools.product(combinations, [True, False] if torch.cuda.is_available() else [False], [True, False] if has_relu else [False])
    for (conv_fn, example_inputs), is_cuda, relu_is_inplace in combinations:
        pattern = get_pattern(conv_fn, relu_is_inplace)
        pattern = get_aten_graph_module(pattern, example_inputs, is_cuda)
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        matches.extend(matcher.match(gm.graph))
    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map['input']
        conv_node = name_node_map['conv']
        weight_node = name_node_map['weight']
        bias_node = name_node_map['bias']
        output_node = name_node_map['output']
        if conv_node.args[0] is not input_node:
            raise ValueError('Conv arg did not contain input node ', input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError('Conv arg did not contain weight node ', weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError('Conv arg did not contain bias node ', bias_node)
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any((not filter_fn(n) for n in partition)):
            continue
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)
        output_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions