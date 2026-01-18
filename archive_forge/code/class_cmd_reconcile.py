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
class cmd_reconcile(Command):
    __doc__ = "Reconcile brz metadata in a branch.\n\n    This can correct data mismatches that may have been caused by\n    previous ghost operations or brz upgrades. You should only\n    need to run this command if 'brz check' or a brz developer\n    advises you to run it.\n\n    If a second branch is provided, cross-branch reconciliation is\n    also attempted, which will check that data like the tree root\n    id which was not present in very early brz versions is represented\n    correctly in both branches.\n\n    At the same time it is run it may recompress data resulting in\n    a potential saving in disk space or performance gain.\n\n    The branch *MUST* be on a listable system such as local disk or sftp.\n    "
    _see_also = ['check']
    takes_args = ['branch?']
    takes_options = [Option('canonicalize-chks', help='Make sure CHKs are in canonical form (repairs bug 522637).', hidden=True)]

    def run(self, branch='.', canonicalize_chks=False):
        from .reconcile import reconcile
        dir = controldir.ControlDir.open(branch)
        reconcile(dir, canonicalize_chks=canonicalize_chks)