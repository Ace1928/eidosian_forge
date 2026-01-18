from __future__ import annotations
import typing as t
from ...completion import (
from ...host_configs import (
from ..compat import (
from ..argparsing.parsers import (
from .value_parsers import (
from .key_value_parsers import (
from .helpers import (
class NetworkRemoteParser(PairParser):
    """Composite argument parser for a network remote host."""

    def create_namespace(self) -> t.Any:
        """Create and return a namespace."""
        return NetworkRemoteConfig()

    def get_left_parser(self, state: ParserState) -> Parser:
        """Return the parser for the left side."""
        names = list(filter_completion(network_completion()))
        for target in state.root_namespace.targets or []:
            names.remove(target.name)
        return NamespaceWrappedParser('name', PlatformParser(names))

    def get_right_parser(self, choice: t.Any) -> Parser:
        """Return the parser for the right side."""
        return NetworkRemoteKeyValueParser()

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        content = '\n'.join([f'  {name}' for name, item in filter_completion(network_completion()).items()])
        content += '\n'.join(['', '  {platform}/{version}  # use an unknown platform and version'])
        state.sections['target remote systems (choose one):'] = content
        return f'{{system}}[,{NetworkRemoteKeyValueParser().document(state)}]'