import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _MissingStandardSymbolicFunction(infra.Rule):
    """Missing symbolic function for standard PyTorch operator, cannot translate node to ONNX."""

    def format_message(self, op_name, opset_version, issue_url) -> str:
        """Returns the formatted default message of this Rule.

        Message template: "Exporting the operator '{op_name}' to ONNX opset version {opset_version} is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: {issue_url}."
        """
        return self.message_default_template.format(op_name=op_name, opset_version=opset_version, issue_url=issue_url)

    def format(self, level: infra.Level, op_name, opset_version, issue_url) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: "Exporting the operator '{op_name}' to ONNX opset version {opset_version} is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub: {issue_url}."
        """
        return (self, level, self.format_message(op_name=op_name, opset_version=opset_version, issue_url=issue_url))