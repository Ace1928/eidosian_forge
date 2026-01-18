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
def _annotate_output_for_int8_in_int8_out_pattern(self, node: Node, quantization_config: QuantizationConfig) -> None:
    """
        Check and insert observer at output of node in int8_in_int8_out_ops_pt2e if needed.
        Recipe refers to https://github.com/intel/intel-extension-for-pytorch/blob/
        90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_utils.py#L495
        """
    edge_or_node: Tuple[Node, Node]
    if node.target in int8_in_int8_out_ops_pt2e and _is_any_annotated([node]):
        if node.target == torch.ops.aten.max_pool2d.default:
            maxpool_node = node
            if not _is_all_annotated([maxpool_node]):
                return
            maxpool_node_quantization_annotation = maxpool_node.meta[QUANT_ANNOTATION_KEY] if QUANT_ANNOTATION_KEY in maxpool_node.meta else None
            if maxpool_node_quantization_annotation and maxpool_node_quantization_annotation._is_output_of_quantized_pattern:
                input_act = maxpool_node.args[0]
                assert isinstance(input_act, Node)
                assert isinstance(maxpool_node, Node)
                edge_or_node = (input_act, maxpool_node)
                maxpool_node_quantization_annotation.output_qspec = SharedQuantizationSpec(edge_or_node)
        else:
            input_node = node.all_input_nodes[0]
            self._annotate_output_share_observer_as_input(input_node, node)
    return