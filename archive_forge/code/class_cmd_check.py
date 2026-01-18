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
class cmd_check(Command):
    __doc__ = "Validate working tree structure, branch consistency and repository history.\n\n    This command checks various invariants about branch and repository storage\n    to detect data corruption or brz bugs.\n\n    The working tree and branch checks will only give output if a problem is\n    detected. The output fields of the repository check are:\n\n    revisions\n        This is just the number of revisions checked.  It doesn't\n        indicate a problem.\n\n    versionedfiles\n        This is just the number of versionedfiles checked.  It\n        doesn't indicate a problem.\n\n    unreferenced ancestors\n        Texts that are ancestors of other texts, but\n        are not properly referenced by the revision ancestry.  This is a\n        subtle problem that Breezy can work around.\n\n    unique file texts\n        This is the total number of unique file contents\n        seen in the checked revisions.  It does not indicate a problem.\n\n    repeated file texts\n        This is the total number of repeated texts seen\n        in the checked revisions.  Texts can be repeated when their file\n        entries are modified, but the file contents are not.  It does not\n        indicate a problem.\n\n    If no restrictions are specified, all data that is found at the given\n    location will be checked.\n\n    :Examples:\n\n        Check the tree and branch at 'foo'::\n\n            brz check --tree --branch foo\n\n        Check only the repository at 'bar'::\n\n            brz check --repo bar\n\n        Check everything at 'baz'::\n\n            brz check baz\n    "
    _see_also = ['reconcile']
    takes_args = ['path?']
    takes_options = ['verbose', Option('branch', help='Check the branch related to the current directory.'), Option('repo', help='Check the repository related to the current directory.'), Option('tree', help='Check the working tree related to the current directory.')]

    def run(self, path=None, verbose=False, branch=False, repo=False, tree=False):
        from .check import check_dwim
        if path is None:
            path = '.'
        if not branch and (not repo) and (not tree):
            branch = repo = tree = True
        check_dwim(path, verbose, do_branch=branch, do_repo=repo, do_tree=tree)