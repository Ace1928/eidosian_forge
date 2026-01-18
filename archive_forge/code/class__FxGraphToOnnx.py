import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _FxGraphToOnnx(infra.Rule):
    """Transforms graph from FX IR to ONNX IR."""

    def format_message(self, graph_name) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'Transforming FX graph {graph_name} to ONNX graph.'
        """
        return self.message_default_template.format(graph_name=graph_name)

    def format(self, level: infra.Level, graph_name) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'Transforming FX graph {graph_name} to ONNX graph.'
        """
        return (self, level, self.format_message(graph_name=graph_name))