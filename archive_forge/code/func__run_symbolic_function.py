from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _run_symbolic_function(graph: _C.Graph, block: _C.Block, node: _C.Node, inputs: Any, env: Dict[_C.Value, _C.Value], operator_export_type=_C_onnx.OperatorExportTypes.ONNX) -> Optional[Union[_C.Value, Sequence[Optional[_C.Value]]]]:
    """Runs a symbolic function.

    The function is used in C++ to export the node to ONNX.

    Returns:
        A single or a tuple of Values.
        None when the node gets cloned as is into the new graph.
    """
    opset_version = GLOBALS.export_onnx_opset_version
    node_kind = node.kind()
    if node_kind.endswith('_'):
        ns_op_name = node_kind[:-1]
    else:
        ns_op_name = node_kind
    namespace, op_name = jit_utils.parse_node_kind(ns_op_name)
    graph_context = jit_utils.GraphContext(graph=graph, block=block, opset=opset_version, original_node=node, params_dict=_params_dict, env=env)
    if _should_aten_fallback(ns_op_name, opset_version, operator_export_type):
        attrs = {k + '_' + node.kindOf(k)[0]: symbolic_helper._node_get(node, k) for k in node.attributeNames()}
        outputs = node.outputsSize()
        attrs['outputs'] = outputs
        return graph_context.aten_op(op_name, *inputs, overload_name=_get_aten_op_overload_name(node), **attrs)
    try:
        if symbolic_helper.is_caffe2_aten_fallback() and opset_version == 9:
            symbolic_caffe2.register_quantized_ops('caffe2', opset_version)
        if namespace == 'quantized' and symbolic_helper.is_caffe2_aten_fallback():
            domain = 'caffe2'
        else:
            domain = namespace
        symbolic_function_name = f'{domain}::{op_name}'
        symbolic_function_group = registration.registry.get_function_group(symbolic_function_name)
        if symbolic_function_group is not None:
            symbolic_fn = symbolic_function_group.get(opset_version)
            if symbolic_fn is not None:
                attrs = {k: symbolic_helper._node_get(node, k) for k in node.attributeNames()}
                return symbolic_fn(graph_context, *inputs, **attrs)
        attrs = {k + '_' + node.kindOf(k)[0]: symbolic_helper._node_get(node, k) for k in node.attributeNames()}
        if namespace == 'onnx':
            return graph_context.op(op_name, *inputs, **attrs, outputs=node.outputsSize())
        raise errors.UnsupportedOperatorError(symbolic_function_name, opset_version, symbolic_function_group.get_min_supported() if symbolic_function_group else None)
    except RuntimeError:
        if operator_export_type == _C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH:
            return None
        elif operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK and (not symbolic_helper.is_caffe2_aten_fallback()):
            attrs = {k + '_' + node.kindOf(k)[0]: symbolic_helper._node_get(node, k) for k in node.attributeNames()}
            return graph_context.aten_op(op_name, *inputs, overload_name=_get_aten_op_overload_name(node), **attrs)
        raise
    except TypeError as e:
        e.args = (f'{e.args[0]} \n(Occurred when translating {op_name}).',)
        raise