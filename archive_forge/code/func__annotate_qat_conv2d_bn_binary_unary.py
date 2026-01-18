import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
def _annotate_qat_conv2d_bn_binary_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
    fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add, torch.nn.ReLU])
    for fused_partition in fused_partitions:
        conv_partition, bn_partition, binary_partition, unary_partition = fused_partition
        conv_node, bn_output_node, binary_node, unary_node = self._get_output_nodes_of_partitions([conv_partition, bn_partition, binary_partition, unary_partition])
        if len(bn_output_node.users) != 1:
            continue
        bn_output_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
        if bn_output_node_idx is None or extra_input_node_idx is None:
            continue
        if bn_output_node != binary_node.args[bn_output_node_idx]:
            raise ValueError(f"{bn_output_node} doesn't match input of binary node")
        extra_input_node = binary_node.args[extra_input_node_idx]
        if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
            continue
        if _is_annotated([unary_node, binary_node, bn_output_node, conv_node]):
            continue
        self._annotate_conv_node_helper(conv_node, False, quantization_config)
        binary_node_input_qspec_map = {}
        binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(quantization_config)
        binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=binary_node_input_qspec_map, _annotated=True)
        unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True, _is_output_of_quantized_pattern=True)
        nodes_to_mark_annotated = list(conv_partition.nodes)
        nodes_to_mark_annotated.extend(list(bn_partition.nodes))
        nodes_to_mark_annotated.extend(list(binary_partition.nodes))
        nodes_to_mark_annotated.extend(list(unary_partition.nodes))
        _mark_nodes_as_annotated(nodes_to_mark_annotated)