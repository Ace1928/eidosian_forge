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
class cmd_cat_revision(Command):
    __doc__ = 'Write out metadata for a revision.\n\n    The revision to print can either be specified by a specific\n    revision identifier, or you can use --revision.\n    '
    hidden = True
    takes_args = ['revision_id?']
    takes_options = ['directory', 'revision']
    encoding = 'strict'

    def print_revision(self, revisions, revid):
        stream = revisions.get_record_stream([(revid,)], 'unordered', True)
        record = next(stream)
        if record.storage_kind == 'absent':
            raise errors.NoSuchRevision(revisions, revid)
        revtext = record.get_bytes_as('fulltext')
        self.outf.write(revtext.decode('utf-8'))

    @display_command
    def run(self, revision_id=None, revision=None, directory='.'):
        if revision_id is not None and revision is not None:
            raise errors.CommandError(gettext('You can only supply one of revision_id or --revision'))
        if revision_id is None and revision is None:
            raise errors.CommandError(gettext('You must supply either --revision or a revision_id'))
        b = controldir.ControlDir.open_containing_tree_or_branch(directory)[1]
        revisions = getattr(b.repository, 'revisions', None)
        if revisions is None:
            raise errors.CommandError(gettext('Repository %r does not support access to raw revision texts') % b.repository)
        with b.repository.lock_read():
            if revision_id is not None:
                revision_id = revision_id.encode('utf-8')
                try:
                    self.print_revision(revisions, revision_id)
                except errors.NoSuchRevision as exc:
                    msg = gettext('The repository {0} contains no revision {1}.').format(b.repository.base, revision_id.decode('utf-8'))
                    raise errors.CommandError(msg) from exc
            elif revision is not None:
                for rev in revision:
                    if rev is None:
                        raise errors.CommandError(gettext('You cannot specify a NULL revision.'))
                    rev_id = rev.as_revision_id(b)
                    self.print_revision(revisions, rev_id)