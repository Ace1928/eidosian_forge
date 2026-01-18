import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _NoSymbolicFunctionForCallFunction(infra.Rule):
    """Cannot find symbolic function to convert the "call_function" FX node to ONNX."""

    def format_message(self, target) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'No symbolic function to convert the "call_function" node {target} to ONNX. '
        """
        return self.message_default_template.format(target=target)

    def format(self, level: infra.Level, target) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'No symbolic function to convert the "call_function" node {target} to ONNX. '
        """
        return (self, level, self.format_message(target=target))