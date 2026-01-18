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
def _annotate_cat(self, node: Node, quantization_config: QuantizationConfig) -> None:
    cat_node = node
    input_nodes = cat_node.args[0]
    assert isinstance(input_nodes, Sequence)
    first_input_node = input_nodes[0]
    input_qspec_map = {}
    assert isinstance(first_input_node, Node)
    assert isinstance(cat_node, Node)
    input_qspec_map[first_input_node] = get_input_act_qspec(quantization_config)
    share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))
    for input_node in input_nodes[1:]:
        if input_node not in input_qspec_map:
            assert isinstance(input_node, Node)
            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec
    cat_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)