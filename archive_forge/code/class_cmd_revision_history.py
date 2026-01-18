import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_revision_history(Command):
    __doc__ = 'Display the list of revision ids on a branch.'
    _see_also = ['log']
    takes_args = ['location?']
    hidden = True

    @display_command
    def run(self, location='.'):
        branch = Branch.open_containing(location)[0]
        self.enter_context(branch.lock_read())
        graph = branch.repository.get_graph()
        history = list(graph.iter_lefthand_ancestry(branch.last_revision(), [_mod_revision.NULL_REVISION]))
        for revid in reversed(history):
            self.outf.write(revid)
            self.outf.write('\n')