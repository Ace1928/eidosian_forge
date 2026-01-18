from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class PosixTargetParser(TargetNamespaceParser, TypeParser):
    """Composite argument parser for a POSIX target."""

    def get_stateless_parsers(self) -> dict[str, Parser]:
        """Return a dictionary of type names and type parsers."""
        parsers: dict[str, Parser] = dict(controller=ControllerParser(), docker=DockerParser(controller=False))
        if get_ci_provider().supports_core_ci_auth():
            parsers.update(remote=PosixRemoteParser(controller=False))
        parsers.update(ssh=PosixSshParser())
        return parsers

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        section = f'{self.option_name} options (choose one):'
        state.sections[section] = ''
        state.sections[section] = '\n'.join([f'  {name}:{parser.document(state)}' for name, parser in self.get_stateless_parsers().items()])
        return None