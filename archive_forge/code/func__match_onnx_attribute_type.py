from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
@_beartype.beartype
def _match_onnx_attribute_type(self, attribute_name: str, attribute: Union[fx_type_utils.Argument, onnxscript_graph_building.TorchScriptTensor], is_sequence: bool=False) -> bool:
    if isinstance(attribute, (int, float, bool, str)):
        attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(type(attribute), is_sequence=is_sequence)
        if attribute_onnx_type != self.attributes[attribute_name].type:
            return False
    elif isinstance(attribute, (list, tuple)) and attribute:
        return self._match_onnx_attribute_type(attribute_name, attribute[0], is_sequence=True)
    else:
        return False
    return True