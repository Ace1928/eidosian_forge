import logging
from typing import Optional
import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import QuantizationSpecBase
from torch.fx.passes.infra.pass_base import PassResult
def _add_metadata(to_node: torch.fx.Node, from_node: torch.fx.Node) -> None:
    from_meta = from_node.meta
    for meta_name in _METADATA_TO_PORT:
        if meta_name in from_meta:
            to_node.meta[meta_name] = from_meta[meta_name]