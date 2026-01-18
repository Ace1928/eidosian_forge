import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def _switch_wc_to_revno(self, revno, outf):
    """Move the working tree to the given revno."""
    self._current.switch(revno)
    self._current.show_rev_log(outf=outf)