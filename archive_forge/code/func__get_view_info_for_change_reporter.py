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
def _get_view_info_for_change_reporter(tree):
    """Get the view information from a tree for change reporting."""
    view_info = None
    try:
        current_view = tree.views.get_view_info()[0]
        if current_view is not None:
            view_info = (current_view, tree.views.lookup_view())
    except views.ViewsNotSupported:
        pass
    return view_info