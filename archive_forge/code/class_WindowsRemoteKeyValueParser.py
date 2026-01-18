from __future__ import annotations
import typing as t
from ...constants import (
from ...completion import (
from ...util import (
from ...host_configs import (
from ...become import (
from ..argparsing.parsers import (
from .value_parsers import (
from .helpers import (
class WindowsRemoteKeyValueParser(KeyValueParser):
    """Composite argument parser for Windows remote key/value pairs."""

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        return dict(provider=ChoicesParser(REMOTE_PROVIDERS), arch=ChoicesParser(REMOTE_ARCHITECTURES))

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        section_name = 'remote options'
        state.sections[f'target {section_name} (comma separated):'] = '\n'.join([f'  provider={ChoicesParser(REMOTE_PROVIDERS).document(state)}', f'  arch={ChoicesParser(REMOTE_ARCHITECTURES).document(state)}'])
        return f'{{{section_name}}}'