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
def _annotate_linear(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
    linear_partitions = get_source_partitions(gm.graph, [torch.nn.Linear, torch.nn.functional.linear])
    linear_partitions = list(itertools.chain(*linear_partitions.values()))
    for partition in linear_partitions:
        if len(partition.output_nodes) > 1:
            raise ValueError('Linear partition cannot have more than one output node')
        linear_node = partition.output_nodes[0]
        if linear_node.op != 'call_function' or linear_node.target not in (torch.ops.aten.linear.default,):
            raise ValueError(f'{linear_node} is not an aten linear operator')
        if _is_annotated([linear_node]):
            continue
        self._annotate_linear_node_helper(linear_node, True, quantization_config)