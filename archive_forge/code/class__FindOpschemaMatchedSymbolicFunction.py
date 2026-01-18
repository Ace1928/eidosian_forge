import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _FindOpschemaMatchedSymbolicFunction(infra.Rule):
    """Find the OnnxFunction that matches the input/attribute dtypes by comparing them with their opschemas."""

    def format_message(self, symbolic_fn, node) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'The OnnxFunction: {symbolic_fn} is the nearest match of the node {node}.'
        """
        return self.message_default_template.format(symbolic_fn=symbolic_fn, node=node)

    def format(self, level: infra.Level, symbolic_fn, node) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'The OnnxFunction: {symbolic_fn} is the nearest match of the node {node}.'
        """
        return (self, level, self.format_message(symbolic_fn=symbolic_fn, node=node))