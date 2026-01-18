from __future__ import annotations
from .argparsing import (
from .parsers import (
class WindowsSshTargetAction(CompositeAction):
    """Composite action parser for a Windows SSH target."""

    def create_parser(self) -> NamespaceParser:
        """Return a namespace parser to parse the argument associated with this action."""
        return WindowsSshTargetParser()