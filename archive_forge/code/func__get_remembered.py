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
def _get_remembered(self, tree, verb_string):
    """Use tree.branch's parent if none was supplied.

        Report if the remembered location was used.
        """
    stored_location = tree.branch.get_submit_branch()
    stored_location_type = 'submit'
    if stored_location is None:
        stored_location = tree.branch.get_parent()
        stored_location_type = 'parent'
    mutter('%s', stored_location)
    if stored_location is None:
        raise errors.CommandError(gettext('No location specified or remembered'))
    display_url = urlutils.unescape_for_display(stored_location, 'utf-8')
    note(gettext('{0} remembered {1} location {2}').format(verb_string, stored_location_type, display_url))
    return stored_location