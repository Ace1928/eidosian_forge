import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def _open_for_read(self):
    """Open log file for reading."""
    if self._filename:
        return self._controldir.control_transport.get(self._filename)
    else:
        return sys.stdin