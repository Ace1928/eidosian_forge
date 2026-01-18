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
class cmd_annotate(Command):
    __doc__ = 'Show the origin of each line in a file.\n\n    This prints out the given file with an annotation on the left side\n    indicating which revision, author and date introduced the change.\n\n    If the origin is the same for a run of consecutive lines, it is\n    shown only at the top, unless the --all option is given.\n    '
    aliases = ['ann', 'blame', 'praise']
    takes_args = ['filename']
    takes_options = [Option('all', help='Show annotations on all lines.'), Option('long', help='Show commit date in annotations.'), 'revision', 'show-ids', 'directory']
    encoding_type = 'exact'

    @display_command
    def run(self, filename, all=False, long=False, revision=None, show_ids=False, directory=None):
        from .annotate import annotate_file_tree
        wt, branch, relpath = _open_directory_or_containing_tree_or_branch(filename, directory)
        if wt is not None:
            self.enter_context(wt.lock_read())
        else:
            self.enter_context(branch.lock_read())
        tree = _get_one_revision_tree('annotate', revision, branch=branch)
        self.enter_context(tree.lock_read())
        if wt is not None and revision is None:
            if not wt.is_versioned(relpath):
                raise errors.NotVersionedError(relpath)
            annotate_file_tree(wt, relpath, self.outf, long, all, show_ids=show_ids)
        else:
            if not tree.is_versioned(relpath):
                raise errors.NotVersionedError(relpath)
            annotate_file_tree(tree, relpath, self.outf, long, all, show_ids=show_ids, branch=branch)