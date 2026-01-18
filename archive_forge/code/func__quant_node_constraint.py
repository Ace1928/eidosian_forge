import torch
from torch.fx import GraphModule
from torch.fx import Node
from .pt2e.prepare import prepare
from .pt2e.qat_utils import (
from .pt2e.utils import (
from .pt2e.representation import reference_representation_rewrite
from .quantize_fx import _convert_to_reference_decomposed_fx
from torch.ao.quantization.quantizer import (  # noqa: F401
from torch.fx.passes.infra.pass_manager import PassManager
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch._inductor.constant_folding import constant_fold
def _quant_node_constraint(n: Node) -> bool:
    """If there is any pure ops between get_attr and quantize op they will be const propagated
    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*
    (Note: dequantize op is not going to be constant propagated)

    This filter is added because we don't want to constant fold the things that are not
    related to quantization
    """
    return n.op == 'call_function' and n.target in _QUANT_OPS