from __future__ import annotations
import abc
import shlex
import tempfile
import typing as t
from .io import (
from .config import (
from .util import (
from .util_common import (
from .docker_util import (
from .ssh import (
from .become import (
class LocalConnection(Connection):
    """Connect to localhost."""

    def __init__(self, args: EnvironmentConfig) -> None:
        self.args = args

    def run(self, command: list[str], capture: bool, interactive: bool=False, data: t.Optional[str]=None, stdin: t.Optional[t.IO[bytes]]=None, stdout: t.Optional[t.IO[bytes]]=None, output_stream: t.Optional[OutputStream]=None) -> tuple[t.Optional[str], t.Optional[str]]:
        """Run the specified command and return the result."""
        return run_command(args=self.args, cmd=command, capture=capture, data=data, stdin=stdin, stdout=stdout, interactive=interactive, output_stream=output_stream)