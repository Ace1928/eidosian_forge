from __future__ import annotations
import textwrap
from typing import Optional
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import diagnostics
class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    def __init__(self, name: str, version: int, supported_version: Optional[int]):
        if supported_version is not None:
            diagnostic_rule: diagnostics.infra.Rule = diagnostics.rules.operator_supported_in_newer_opset_version
            msg = diagnostic_rule.format_message(name, version, supported_version)
            diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
        elif name.startswith(('aten::', 'prim::', 'quantized::')):
            diagnostic_rule = diagnostics.rules.missing_standard_symbolic_function
            msg = diagnostic_rule.format_message(name, version, _constants.PYTORCH_GITHUB_ISSUES_URL)
            diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
        else:
            diagnostic_rule = diagnostics.rules.missing_custom_symbolic_function
            msg = diagnostic_rule.format_message(name)
            diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
        super().__init__(msg)