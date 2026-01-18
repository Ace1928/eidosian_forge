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
class cmd_split(Command):
    __doc__ = "Split a subdirectory of a tree into a separate tree.\n\n    This command will produce a target tree in a format that supports\n    rich roots, like 'rich-root' or 'rich-root-pack'.  These formats cannot be\n    converted into earlier formats like 'dirstate-tags'.\n\n    The TREE argument should be a subdirectory of a working tree.  That\n    subdirectory will be converted into an independent tree, with its own\n    branch.  Commits in the top-level tree will not apply to the new subtree.\n    "
    _see_also = ['join']
    takes_args = ['tree']

    def run(self, tree):
        containing_tree, subdir = WorkingTree.open_containing(tree)
        if not containing_tree.is_versioned(subdir):
            raise errors.NotVersionedError(subdir)
        try:
            containing_tree.extract(subdir)
        except errors.RootNotRich as exc:
            raise errors.RichRootUpgradeRequired(containing_tree.branch.base) from exc