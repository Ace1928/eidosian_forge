from __future__ import annotations
from .argparsing import (
from .parsers import (
class SanityPythonTargetAction(CompositeAction):
    """Composite action parser for a sanity target."""

    def create_parser(self) -> NamespaceParser:
        """Return a namespace parser to parse the argument associated with this action."""
        return SanityPythonTargetParser()