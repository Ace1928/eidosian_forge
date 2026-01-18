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
class cmd_link_tree(Command):
    __doc__ = 'Hardlink matching files to another tree.\n\n    Only files with identical content and execute bit will be linked.\n    '
    takes_args = ['location']

    def run(self, location):
        from .transform import link_tree
        target_tree = WorkingTree.open_containing('.')[0]
        source_tree = WorkingTree.open(location)
        with target_tree.lock_write(), source_tree.lock_read():
            link_tree(target_tree, source_tree)