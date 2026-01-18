from __future__ import annotations
import glob
import io
import os
import shutil
import zipfile
from typing import Any, List, Mapping, Set, Tuple, Union
import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _find_onnxscript_op(graph_proto, included_node_func: Set[str], custom_opsets: Mapping[str, int], onnx_function_list: List):
    """Recursively iterate ModelProto to find ONNXFunction op as it may contain control flow Op."""
    for node in graph_proto.node:
        node_kind = node.domain + '::' + node.op_type
        for attr in node.attribute:
            if attr.g is not None:
                _find_onnxscript_op(attr.g, included_node_func, custom_opsets, onnx_function_list)
        onnx_function_group = registration.registry.get_function_group(node_kind)
        if node.domain and (not jit_utils.is_aten(node.domain)) and (not jit_utils.is_prim(node.domain)) and (not jit_utils.is_onnx(node.domain)) and (onnx_function_group is not None) and (node_kind not in included_node_func):
            specified_version = custom_opsets.get(node.domain, 1)
            onnx_fn = onnx_function_group.get(specified_version)
            if onnx_fn is not None:
                if hasattr(onnx_fn, 'to_function_proto'):
                    onnx_function_proto = onnx_fn.to_function_proto()
                    onnx_function_list.append(onnx_function_proto)
                    included_node_func.add(node_kind)
                continue
            raise errors.UnsupportedOperatorError(node_kind, specified_version, onnx_function_group.get_min_supported() if onnx_function_group else None)
    return (onnx_function_list, included_node_func)