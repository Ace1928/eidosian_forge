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
class DockerKeyValueParser(KeyValueParser):
    """Composite argument parser for docker key/value pairs."""

    def __init__(self, image: str, controller: bool) -> None:
        self.controller = controller
        self.versions = get_docker_pythons(image, controller, False)
        self.allow_default = bool(get_docker_pythons(image, controller, True))

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        return dict(python=PythonParser(versions=self.versions, allow_venv=False, allow_default=self.allow_default), seccomp=ChoicesParser(SECCOMP_CHOICES), cgroup=EnumValueChoicesParser(CGroupVersion), audit=EnumValueChoicesParser(AuditMode), privileged=BooleanParser(), memory=IntegerParser())

    def document(self, state: DocumentationState) -> t.Optional[str]:
        """Generate and return documentation for this parser."""
        python_parser = PythonParser(versions=[], allow_venv=False, allow_default=self.allow_default)
        section_name = 'docker options'
        state.sections[f'{('controller' if self.controller else 'target')} {section_name} (comma separated):'] = '\n'.join([f'  python={python_parser.document(state)}', f'  seccomp={ChoicesParser(SECCOMP_CHOICES).document(state)}', f'  cgroup={EnumValueChoicesParser(CGroupVersion).document(state)}', f'  audit={EnumValueChoicesParser(AuditMode).document(state)}', f'  privileged={BooleanParser().document(state)}', f'  memory={IntegerParser().document(state)}  # bytes'])
        return f'{{{section_name}}}'