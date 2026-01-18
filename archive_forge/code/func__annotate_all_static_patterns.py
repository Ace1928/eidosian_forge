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
def _annotate_all_static_patterns(self, model: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> torch.fx.GraphModule:
    if quantization_config is None:
        return model
    if quantization_config.is_qat:
        for op in self.STATIC_QAT_ONLY_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
    for op in self.STATIC_OPS:
        OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
    return model