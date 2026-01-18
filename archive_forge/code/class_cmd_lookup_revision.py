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
class cmd_lookup_revision(Command):
    __doc__ = 'Lookup the revision-id from a revision-number\n\n    :Examples:\n        brz lookup-revision 33\n    '
    hidden = True
    takes_args = ['revno']
    takes_options = ['directory']

    @display_command
    def run(self, revno, directory='.'):
        try:
            revno = int(revno)
        except ValueError as exc:
            raise errors.CommandError(gettext('not a valid revision-number: %r') % revno) from exc
        revid = WorkingTree.open_containing(directory)[0].branch.get_rev_id(revno)
        self.outf.write('%s\n' % revid.decode('utf-8'))