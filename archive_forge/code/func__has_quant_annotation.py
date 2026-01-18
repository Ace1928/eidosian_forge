import logging
from typing import Optional
import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import QuantizationSpecBase
from torch.fx.passes.infra.pass_base import PassResult
def _has_quant_annotation(node: torch.fx.Node) -> bool:
    return 'quantization_annotation' in node.meta