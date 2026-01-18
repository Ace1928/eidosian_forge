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
class cmd_checkout(Command):
    __doc__ = 'Create a new checkout of an existing branch.\n\n    If BRANCH_LOCATION is omitted, checkout will reconstitute a working tree\n    for the branch found in \'.\'. This is useful if you have removed the working\n    tree or if it was never created - i.e. if you pushed the branch to its\n    current location using SFTP.\n\n    If the TO_LOCATION is omitted, the last component of the BRANCH_LOCATION\n    will be used.  In other words, "checkout ../foo/bar" will attempt to create\n    ./bar.  If the BRANCH_LOCATION has no / or path separator embedded, the\n    TO_LOCATION is derived from the BRANCH_LOCATION by stripping a leading\n    scheme or drive identifier, if any. For example, "checkout lp:foo-bar" will\n    attempt to create ./foo-bar.\n\n    To retrieve the branch as of a particular revision, supply the --revision\n    parameter, as in "checkout foo/bar -r 5". Note that this will be\n    immediately out of date [so you cannot commit] but it may be useful (i.e.\n    to examine old code.)\n    '
    _see_also = ['checkouts', 'branch', 'working-trees', 'remove-tree']
    takes_args = ['branch_location?', 'to_location?']
    takes_options = ['revision', Option('lightweight', help='Perform a lightweight checkout.  Lightweight checkouts depend on access to the branch for every operation.  Normal checkouts can perform common operations like diff and status without such access, and also support local commits.'), Option('files-from', type=str, help='Get file contents from this tree.'), Option('hardlink', help='Hard-link working tree files where possible.')]
    aliases = ['co']

    def run(self, branch_location=None, to_location=None, revision=None, lightweight=False, files_from=None, hardlink=False):
        if branch_location is None:
            branch_location = osutils.getcwd()
            to_location = branch_location
        accelerator_tree, source = controldir.ControlDir.open_tree_or_branch(branch_location)
        if not (hardlink or files_from):
            accelerator_tree = None
        revision = _get_one_revision('checkout', revision)
        if files_from is not None and files_from != branch_location:
            accelerator_tree = WorkingTree.open(files_from)
        if revision is not None:
            revision_id = revision.as_revision_id(source)
        else:
            revision_id = None
        if to_location is None:
            to_location = urlutils.derive_to_location(branch_location)
        if osutils.abspath(to_location) == osutils.abspath(branch_location):
            try:
                source.controldir.open_workingtree()
            except errors.NoWorkingTree:
                source.controldir.create_workingtree(revision_id)
                return
        source.create_checkout(to_location, revision_id=revision_id, lightweight=lightweight, accelerator_tree=accelerator_tree, hardlink=hardlink)