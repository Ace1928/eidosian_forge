import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _FindOperatorOverloadsInOnnxRegistry(infra.Rule):
    """Find the list of OnnxFunction of the PyTorch operator in onnx registry."""

    def format_message(self, node) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'Checking if the FX node: {node} is supported in onnx registry.'
        """
        return self.message_default_template.format(node=node)

    def format(self, level: infra.Level, node) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'Checking if the FX node: {node} is supported in onnx registry.'
        """
        return (self, level, self.format_message(node=node))