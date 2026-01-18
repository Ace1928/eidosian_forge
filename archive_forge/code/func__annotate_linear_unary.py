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
def _annotate_linear_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
    postop_list = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Tanh]
    fused_partitions: List[tuple] = []
    for postop in postop_list:
        fused_partitions = fused_partitions + find_sequential_partitions(gm, [torch.nn.Linear, postop])
    for fused_partition in fused_partitions:
        linear_partition, unary_partition = fused_partition
        linear_node, unary_node = self._get_output_nodes_of_partitions([linear_partition, unary_partition])
        if linear_node.op != 'call_function' or linear_node.target not in (torch.ops.aten.linear.default,):
            continue
        if _is_annotated([unary_node, linear_node]):
            continue
        self._annotate_linear_node_helper(linear_node, False, quantization_config)
        unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(_annotated=True, _is_output_of_quantized_pattern=True)