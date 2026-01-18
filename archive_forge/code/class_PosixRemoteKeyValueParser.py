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
class PosixRemoteKeyValueParser(KeyValueParser):
    """Composite argument parser for POSIX remote key/value pairs."""

    def __init__(self, name: str, controller: bool) -> None:
        self.controller = controller
        self.versions = get_remote_pythons(name, controller, False)
        self.allow_default = bool(get_remote_pythons(name, controller, True))

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        return dict(become=ChoicesParser(list(SUPPORTED_BECOME_METHODS)), provider=ChoicesParser(REMOTE_PROVIDERS), arch=ChoicesParser(REMOTE_ARCHITECTURES), python=PythonParser(versions=self.versions, allow_venv=False, allow_default=self.allow_default))

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        python_parser = PythonParser(versions=[], allow_venv=False, allow_default=self.allow_default)
        section_name = 'remote options'
        state.sections[f'{('controller' if self.controller else 'target')} {section_name} (comma separated):'] = '\n'.join([f'  become={ChoicesParser(list(SUPPORTED_BECOME_METHODS)).document(state)}', f'  provider={ChoicesParser(REMOTE_PROVIDERS).document(state)}', f'  arch={ChoicesParser(REMOTE_ARCHITECTURES).document(state)}', f'  python={python_parser.document(state)}'])
        return f'{{{section_name}}}'