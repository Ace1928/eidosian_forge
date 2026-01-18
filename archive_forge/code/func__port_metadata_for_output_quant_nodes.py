import logging
from typing import Optional
import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import QuantizationSpecBase
from torch.fx.passes.infra.pass_base import PassResult
def _port_metadata_for_output_quant_nodes(node: torch.fx.Node, qspec: Optional[QuantizationSpecBase]):
    if qspec is None:
        return
    node_users = _filter_sym_size_users(node)
    if len(node_users) != 1:
        raise InternalError(f'Expecting {node} to have single user')
    q_node = node_users.pop()
    if q_node.op != 'call_function' or q_node.target not in _QUANTIZE_OPS:
        logger.warning(f'Expecting {node} user to be a quantized op but got {q_node}')
        return
    _add_metadata(q_node, node)