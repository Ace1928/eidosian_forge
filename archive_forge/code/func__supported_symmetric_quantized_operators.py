from __future__ import annotations
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
def _supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {'conv2d': [[torch.nn.Conv2d, torch.nn.ReLU], [torch.nn.Conv2d, F.relu], [F.conv2d, torch.nn.ReLU], [F.conv2d, F.relu]], 'linear': [[torch.nn.Linear], [F.linear]], 'add': [[torch.add]], 'max_pool2d': [[torch.nn.MaxPool2d], [F.max_pool2d]], 'adaptive_avg_pool2d': [[torch.nn.AdaptiveAvgPool2d], [F.adaptive_avg_pool2d]]}
    return copy.deepcopy(supported_operators)