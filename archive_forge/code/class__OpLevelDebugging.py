import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _OpLevelDebugging(infra.Rule):
    """Report any op level validation failure in warnings."""

    def format_message(self, node, symbolic_fn) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'FX node: {node} and its onnx function: {symbolic_fn} fails on op level validation.'
        """
        return self.message_default_template.format(node=node, symbolic_fn=symbolic_fn)

    def format(self, level: infra.Level, node, symbolic_fn) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'FX node: {node} and its onnx function: {symbolic_fn} fails on op level validation.'
        """
        return (self, level, self.format_message(node=node, symbolic_fn=symbolic_fn))