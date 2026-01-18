from __future__ import annotations
import typing as t
from ...completion import (
from ...host_configs import (
from ..compat import (
from ..argparsing.parsers import (
from .value_parsers import (
from .key_value_parsers import (
from .helpers import (
class PosixRemoteParser(PairParser):
    """Composite argument parser for a POSIX remote host."""

    def __init__(self, controller: bool) -> None:
        self.controller = controller

    def create_namespace(self) -> t.Any:
        """Create and return a namespace."""
        return PosixRemoteConfig()

    def get_left_parser(self, state: ParserState) -> Parser:
        """Return the parser for the left side."""
        return NamespaceWrappedParser('name', PlatformParser(list(filter_completion(remote_completion(), controller_only=self.controller))))

    def get_right_parser(self, choice: t.Any) -> Parser:
        """Return the parser for the right side."""
        return PosixRemoteKeyValueParser(choice, self.controller)

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        value: PosixRemoteConfig = super().parse(state)
        if not value.python and (not get_remote_pythons(value.name, self.controller, True)):
            raise ParserError(f'Python version required for remote: {value.name}')
        return value

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        default = get_fallback_remote_controller()
        content = '\n'.join([f'  {name} ({', '.join(get_remote_pythons(name, self.controller, False))})' for name, item in filter_completion(remote_completion(), controller_only=self.controller).items()])
        content += '\n'.join(['', '  {platform}/{version}  # python must be specified for unknown systems'])
        state.sections[f'{('controller' if self.controller else 'target')} remote systems and supported python versions (choose one):'] = content
        return f'{{system}}[,{PosixRemoteKeyValueParser(default, self.controller).document(state)}]'