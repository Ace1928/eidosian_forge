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
def _select_branch_location(self, tree, user_location, revision=None, index=None):
    """Select a branch location, according to possible inputs.

        If provided, branches from ``revision`` are preferred.  (Both
        ``revision`` and ``index`` must be supplied.)

        Otherwise, the ``location`` parameter is used.  If it is None, then the
        ``submit`` or ``parent`` location is used, and a note is printed.

        :param tree: The working tree to select a branch for merging into
        :param location: The location entered by the user
        :param revision: The revision parameter to the command
        :param index: The index to use for the revision parameter.  Negative
            indices are permitted.
        :return: (selected_location, user_location).  The default location
            will be the user-entered location.
        """
    if revision is not None and index is not None and (revision[index] is not None):
        branch = revision[index].get_branch()
        if branch is not None:
            return (branch, branch)
    if user_location is None:
        location = self._get_remembered(tree, 'Merging from')
    else:
        location = user_location
    return (location, user_location)