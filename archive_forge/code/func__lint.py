from __future__ import annotations
import dataclasses
from typing import Dict
from torch.onnx._internal.fx import _pass, diagnostics, registration
def _lint(self, analysis_result: UnsupportedFxNodesAnalysisResult, diagnostic_level: diagnostics.infra.Level):
    """Lint the graph and emit diagnostics if unsupported FX nodes are found."""
    if not analysis_result.unsupported_op_to_target_mapping:
        return
    normalized_op_targets_map = {op: list(targets.keys()) for op, targets in analysis_result.unsupported_op_to_target_mapping.items()}
    rule = diagnostics.rules.unsupported_fx_node_analysis
    diagnostic = diagnostics.Diagnostic(rule, level=diagnostic_level, message=rule.format_message(normalized_op_targets_map))
    self.diagnostic_context.log_and_raise_if_error(diagnostic)