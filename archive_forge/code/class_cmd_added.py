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
class cmd_added(Command):
    __doc__ = 'List files added in working tree.\n    '
    hidden = True
    _see_also = ['status', 'ls']
    takes_options = ['directory', 'null']

    @display_command
    def run(self, null=False, directory='.'):
        wt = WorkingTree.open_containing(directory)[0]
        self.enter_context(wt.lock_read())
        basis = wt.basis_tree()
        self.enter_context(basis.lock_read())
        for path in wt.all_versioned_paths():
            if basis.has_filename(path):
                continue
            if path == '':
                continue
            if not os.access(osutils.pathjoin(wt.basedir, path), os.F_OK):
                continue
            if null:
                self.outf.write(path + '\x00')
            else:
                self.outf.write(osutils.quotefn(path) + '\n')