from __future__ import annotations
import typing as t
from ...completion import (
from ...host_configs import (
from ..compat import (
from ..argparsing.parsers import (
from .value_parsers import (
from .key_value_parsers import (
from .helpers import (
class PosixSshParser(PairParser):
    """Composite argument parser for a POSIX SSH host."""

    def create_namespace(self) -> t.Any:
        """Create and return a namespace."""
        return PosixSshConfig()

    def get_left_parser(self, state: ParserState) -> Parser:
        """Return the parser for the left side."""
        return SshConnectionParser()

    def get_right_parser(self, choice: t.Any) -> Parser:
        """Return the parser for the right side."""
        return PosixSshKeyValueParser()

    @property
    def required(self) -> bool:
        """True if the delimiter (and thus right parser) is required, otherwise False."""
        return True

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        return f'{SshConnectionParser().document(state)}[,{PosixSshKeyValueParser().document(state)}]'