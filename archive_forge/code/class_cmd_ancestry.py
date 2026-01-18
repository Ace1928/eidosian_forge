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
class cmd_ancestry(Command):
    __doc__ = 'List all revisions merged into this branch.'
    _see_also = ['log', 'revision-history']
    takes_args = ['location?']
    hidden = True

    @display_command
    def run(self, location='.'):
        try:
            wt = WorkingTree.open_containing(location)[0]
        except errors.NoWorkingTree:
            b = Branch.open(location)
            last_revision = b.last_revision()
        else:
            b = wt.branch
            last_revision = wt.last_revision()
        self.enter_context(b.repository.lock_read())
        graph = b.repository.get_graph()
        revisions = [revid for revid, parents in graph.iter_ancestry([last_revision])]
        for revision_id in reversed(revisions):
            if _mod_revision.is_null(revision_id):
                continue
            self.outf.write(revision_id.decode('utf-8') + '\n')