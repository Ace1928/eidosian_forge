import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _MissingCustomSymbolicFunction(infra.Rule):
    """Missing symbolic function for custom PyTorch operator, cannot translate node to ONNX."""

    def format_message(self, op_name) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'ONNX export failed on an operator with unrecognized namespace {op_name}. If you are trying to export a custom operator, make sure you registered it with the right domain and version.'
        """
        return self.message_default_template.format(op_name=op_name)

    def format(self, level: infra.Level, op_name) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'ONNX export failed on an operator with unrecognized namespace {op_name}. If you are trying to export a custom operator, make sure you registered it with the right domain and version.'
        """
        return (self, level, self.format_message(op_name=op_name))