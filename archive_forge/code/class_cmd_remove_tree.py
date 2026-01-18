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
class cmd_remove_tree(Command):
    __doc__ = 'Remove the working tree from a given branch/checkout.\n\n    Since a lightweight checkout is little more than a working tree\n    this will refuse to run against one.\n\n    To re-create the working tree, use "brz checkout".\n    '
    _see_also = ['checkout', 'working-trees']
    takes_args = ['location*']
    takes_options = [Option('force', help='Remove the working tree even if it has uncommitted or shelved changes.')]

    def run(self, location_list, force=False):
        if not location_list:
            location_list = ['.']
        for location in location_list:
            d = controldir.ControlDir.open(location)
            try:
                working = d.open_workingtree()
            except errors.NoWorkingTree as exc:
                raise errors.CommandError(gettext('No working tree to remove')) from exc
            except errors.NotLocalUrl as exc:
                raise errors.CommandError(gettext('You cannot remove the working tree of a remote path')) from exc
            if not force:
                if working.has_changes():
                    raise errors.UncommittedChanges(working)
                if working.get_shelf_manager().last_shelf() is not None:
                    raise errors.ShelvedChanges(working)
            if working.user_url != working.branch.user_url:
                raise errors.CommandError(gettext('You cannot remove the working tree from a lightweight checkout'))
            d.destroy_workingtree()