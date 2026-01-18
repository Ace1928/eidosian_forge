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
class cmd_repair_workingtree(Command):
    __doc__ = 'Reset the working tree state file.\n\n    This is not meant to be used normally, but more as a way to recover from\n    filesystem corruption, etc. This rebuilds the working inventory back to a\n    \'known good\' state. Any new modifications (adding a file, renaming, etc)\n    will be lost, though modified files will still be detected as such.\n\n    Most users will want something more like "brz revert" or "brz update"\n    unless the state file has become corrupted.\n\n    By default this attempts to recover the current state by looking at the\n    headers of the state file. If the state file is too corrupted to even do\n    that, you can supply --revision to force the state of the tree.\n    '
    takes_options = ['revision', 'directory', Option('force', help="Reset the tree even if it doesn't appear to be corrupted.")]
    hidden = True

    def run(self, revision=None, directory='.', force=False):
        tree, _ = WorkingTree.open_containing(directory)
        self.enter_context(tree.lock_tree_write())
        if not force:
            try:
                tree.check_state()
            except errors.BzrError:
                pass
            else:
                raise errors.CommandError(gettext('The tree does not appear to be corrupt. You probably want "brz revert" instead. Use "--force" if you are sure you want to reset the working tree.'))
        if revision is None:
            revision_ids = None
        else:
            revision_ids = [r.as_revision_id(tree.branch) for r in revision]
        try:
            tree.reset_state(revision_ids)
        except errors.BzrError as exc:
            if revision_ids is None:
                extra = gettext(', the header appears corrupt, try passing -r -1 to set the state to the last commit')
            else:
                extra = ''
            raise errors.CommandError(gettext('failed to reset the tree state{0}').format(extra)) from exc