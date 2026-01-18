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
def _is_colocated(control_dir, possible_transports=None):
    """Check if the branch in control_dir is colocated.

    :param control_dir: Control directory
    :return: Tuple with boolean indicating whether the branch is colocated
        and the full URL to the actual branch
    """
    this_url = _get_branch_location(control_dir, possible_transports=possible_transports)
    try:
        root = controldir.ControlDir.open(this_url, possible_transports=possible_transports)
    except errors.NotBranchError:
        return (False, this_url)
    else:
        try:
            control_dir.open_workingtree()
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            return (False, this_url)
        else:
            return (root._format.colocated_branches and control_dir.control_url == root.control_url, this_url)