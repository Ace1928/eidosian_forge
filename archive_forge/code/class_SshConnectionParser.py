from __future__ import annotations
import collections.abc as c
import typing as t
from ...host_configs import (
from ..argparsing.parsers import (
class SshConnectionParser(Parser):
    """
    Composite argument parser for connecting to a host using SSH.
    Format: user@host[:port]
    """
    EXPECTED_FORMAT = '{user}@{host}[:{port}]'

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        namespace = state.current_namespace
        with state.delimit('@'):
            user = AnyParser(no_match_message=f'Expected {{user}} from: {self.EXPECTED_FORMAT}').parse(state)
        setattr(namespace, 'user', user)
        with state.delimit(':', required=False) as colon:
            host = AnyParser(no_match_message=f'Expected {{host}} from: {self.EXPECTED_FORMAT}').parse(state)
        setattr(namespace, 'host', host)
        if colon.match:
            port = IntegerParser(65535).parse(state)
            setattr(namespace, 'port', port)
        return namespace

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        return self.EXPECTED_FORMAT