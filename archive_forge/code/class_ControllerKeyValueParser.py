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
class ControllerKeyValueParser(KeyValueParser):
    """Composite argument parser for controller key/value pairs."""

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        versions = get_controller_pythons(state.root_namespace.controller, False)
        allow_default = bool(get_controller_pythons(state.root_namespace.controller, True))
        allow_venv = isinstance(state.root_namespace.controller, OriginConfig) or not state.root_namespace.controller
        return dict(python=PythonParser(versions=versions, allow_venv=allow_venv, allow_default=allow_default))

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        section_name = 'controller options'
        state.sections[f'target {section_name} (comma separated):'] = '\n'.join([f'  python={PythonParser(SUPPORTED_PYTHON_VERSIONS, allow_venv=False, allow_default=True).document(state)}  # non-origin controller', f'  python={PythonParser(SUPPORTED_PYTHON_VERSIONS, allow_venv=True, allow_default=True).document(state)}  # origin controller'])
        return f'{{{section_name}}}  # default'