from __future__ import annotations
from typing import Callable, Dict, Set, Union
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import registration
@_beartype.beartype
def _create_onnx_supports_op_overload_table(registry) -> Set[Union[torch._ops.OperatorBase, Callable]]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry (OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    table: Set[Union[torch._ops.OperatorBase, Callable]] = set()
    onnx_supported_aten_lookup_table = [k.split('::')[1].split('.')[0] for k in registry._all_registered_ops() if k.startswith('aten::')]
    for op_namespace in (torch.ops.aten, torch.ops.prims):
        attr_names = dir(op_namespace)
        if op_namespace is torch.ops.aten:
            attr_names += onnx_supported_aten_lookup_table
        for attr_name in attr_names:
            if not hasattr(op_namespace, attr_name):
                continue
            op_overload_packet = getattr(op_namespace, attr_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue
            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                internal_op_name = registration.OpName.from_qualified_name(qualified_name=op_overload.name())
                if registry.is_registered_op(namespace=internal_op_name.namespace, op_name=internal_op_name.op_name, overload=internal_op_name.overload) or registry.is_registered_op(namespace=internal_op_name.namespace, op_name=internal_op_name.op_name, overload=None):
                    table.add(op_overload)
    return table