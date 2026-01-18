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
class OriginKeyValueParser(KeyValueParser):
    """Composite argument parser for origin key/value pairs."""

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        versions = CONTROLLER_PYTHON_VERSIONS
        return dict(python=PythonParser(versions=versions, allow_venv=True, allow_default=True))

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        python_parser = PythonParser(versions=CONTROLLER_PYTHON_VERSIONS, allow_venv=True, allow_default=True)
        section_name = 'origin options'
        state.sections[f'controller {section_name} (comma separated):'] = '\n'.join([f'  python={python_parser.document(state)}'])
        return f'{{{section_name}}}  # default'