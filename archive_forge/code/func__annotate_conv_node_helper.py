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
def _annotate_conv_node_helper(self, conv_node: torch.fx.Node, annotate_output: bool, quantization_config: QuantizationConfig) -> None:
    """Helper function to annotate the conv node"""
    input_qspec_map = {}
    input_node = conv_node.args[0]
    assert isinstance(input_node, Node)
    input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
    weight_node = conv_node.args[1]
    assert isinstance(weight_node, Node)
    input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
    bias_node = None if len(conv_node.args) == 2 else conv_node.args[2]
    if isinstance(bias_node, Node):
        input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
    if annotate_output:
        conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)
    else:
        conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)