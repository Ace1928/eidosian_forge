from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
class PythonTargetParser(TargetsNamespaceParser, Parser):
    """Composite argument parser for a Python target."""

    def __init__(self, allow_venv: bool) -> None:
        super().__init__()
        self.allow_venv = allow_venv

    @property
    def option_name(self) -> str:
        """The option name used for this parser."""
        return '--target-python'

    def get_value(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result, without storing the result in the namespace."""
        versions = list(SUPPORTED_PYTHON_VERSIONS)
        for target in state.root_namespace.targets or []:
            versions.remove(target.python.version)
        parser = PythonParser(versions, allow_venv=self.allow_venv, allow_default=True)
        python = parser.parse(state)
        value = ControllerConfig(python=python)
        return value

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        section = f'{self.option_name} options (choose one):'
        state.sections[section] = '\n'.join([f'  {PythonParser(SUPPORTED_PYTHON_VERSIONS, allow_venv=False, allow_default=True).document(state)}  # non-origin controller', f'  {PythonParser(SUPPORTED_PYTHON_VERSIONS, allow_venv=True, allow_default=True).document(state)}  # origin controller'])
        return None