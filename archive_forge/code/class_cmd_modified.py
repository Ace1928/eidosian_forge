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
class cmd_modified(Command):
    __doc__ = 'List files modified in working tree.\n    '
    hidden = True
    _see_also = ['status', 'ls']
    takes_options = ['directory', 'null']

    @display_command
    def run(self, null=False, directory='.'):
        tree = WorkingTree.open_containing(directory)[0]
        self.enter_context(tree.lock_read())
        td = tree.changes_from(tree.basis_tree())
        self.cleanup_now()
        for change in td.modified:
            if null:
                self.outf.write(change.path[1] + '\x00')
            else:
                self.outf.write(osutils.quotefn(change.path[1]) + '\n')