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
def _get_input_idx_for_binary_node(self, conv_gemm_node: torch.fx.Node, binary_node: torch.fx.Node):
    """Helper function to check conv_gemm and extra input node index
        for binary node fused with conv_gemm.
        """
    conv_gemm_node_idx = None
    extra_input_node_idx = None
    if binary_node.args[0].op == 'call_function' and binary_node.args[0] == conv_gemm_node:
        conv_gemm_node_idx = 0
        extra_input_node_idx = 1
    elif binary_node.args[1].op == 'call_function' and binary_node.args[1] == conv_gemm_node:
        conv_gemm_node_idx = 1
        extra_input_node_idx = 0
    extra_input_node = binary_node.args[extra_input_node_idx]
    assert isinstance(extra_input_node, Node)
    return (conv_gemm_node_idx, extra_input_node_idx)