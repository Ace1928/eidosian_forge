from __future__ import annotations
import typing as t
from ...completion import (
from ...host_configs import (
from ..compat import (
from ..argparsing.parsers import (
from .value_parsers import (
from .key_value_parsers import (
from .helpers import (
class WindowsInventoryParser(PairParser):
    """Composite argument parser for a Windows inventory."""

    def create_namespace(self) -> t.Any:
        """Create and return a namespace."""
        return WindowsInventoryConfig()

    def get_left_parser(self, state: ParserState) -> Parser:
        """Return the parser for the left side."""
        return NamespaceWrappedParser('path', FileParser())

    def get_right_parser(self, choice: t.Any) -> Parser:
        """Return the parser for the right side."""
        return EmptyKeyValueParser()

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        return '{path}  # INI format inventory file'