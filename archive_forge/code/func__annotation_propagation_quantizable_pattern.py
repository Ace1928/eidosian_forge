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
def _annotation_propagation_quantizable_pattern(self, node: Node, quantization_config: QuantizationConfig) -> None:
    if node.target in quantizable_ops_pt2e and (not _is_any_annotated([node])) and (node.op == 'call_function'):

        def is_all_inputs_connected_to_quantized_op(input_nodes):
            for input_node in input_nodes:
                if not _is_quantized_op_pt2e(input_node):
                    return False
            return True
        if node.target is torch.ops.aten.max_pool2d.default:
            input_nodes_to_check = [node.all_input_nodes[0]]
            if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                return
            self._annotate_maxpool2d(node, quantization_config)
            return
        elif node.target is torch.ops.aten.cat.default:
            input_nodes_to_check = node.all_input_nodes
            if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                return
            self._annotate_cat(node, quantization_config)
        else:
            input_node = node.all_input_nodes[0]
            if not is_all_inputs_connected_to_quantized_op([input_node]):
                return
            input_qspec_map = {}
            input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)
    return