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
class PosixSshKeyValueParser(KeyValueParser):
    """Composite argument parser for POSIX SSH host key/value pairs."""

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        return dict(python=PythonParser(versions=list(SUPPORTED_PYTHON_VERSIONS), allow_venv=False, allow_default=False))

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        python_parser = PythonParser(versions=SUPPORTED_PYTHON_VERSIONS, allow_venv=False, allow_default=False)
        section_name = 'ssh options'
        state.sections[f'target {section_name} (comma separated):'] = '\n'.join([f'  python={python_parser.document(state)}'])
        return f'{{{section_name}}}'