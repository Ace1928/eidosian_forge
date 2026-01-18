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
def _supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {'conv2d': [[torch.nn.Conv2d], [F.conv2d]]}
    conv_add_relu_options = itertools.product([torch.nn.Conv2d, F.conv2d], [torch.add, operator.add, None], [torch.nn.ReLU, F.relu, None])
    for conv_op, add_op, relu_op in conv_add_relu_options:
        if add_op is None:
            supported_operators['conv2d'].append([conv_op, relu_op])
        elif relu_op is None:
            supported_operators['conv2d'].append([conv_op, add_op])
        else:
            supported_operators['conv2d'].append([conv_op, add_op, relu_op])
    return copy.deepcopy(supported_operators)