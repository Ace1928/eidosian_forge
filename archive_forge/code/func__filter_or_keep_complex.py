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
def _filter_or_keep_complex(self, node, default_and_custom_functions: List[registration.ONNXFunction], diagnostic_context: diagnostics.DiagnosticContext) -> List[registration.ONNXFunction]:
    if any((torch.is_complex(arg.meta['val']) for arg in node.args if isinstance(arg, torch.fx.Node) and 'val' in arg.meta and isinstance(arg.meta['val'], torch.Tensor))):
        default_and_custom_functions = [func for func in default_and_custom_functions if func.is_complex]
        if not default_and_custom_functions:
            op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
            diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find any COMPLEX symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
            diagnostic_context.log(diagnostic)
            raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
    else:
        default_and_custom_functions = [func for func in default_and_custom_functions if not func.is_complex]
        if not default_and_custom_functions:
            op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
            diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Can ONLY find COMPLEX symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
            diagnostic_context.log(diagnostic)
            raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
    return default_and_custom_functions