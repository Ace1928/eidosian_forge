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
class cmd_break_lock(Command):
    __doc__ = "Break a dead lock.\n\n    This command breaks a lock on a repository, branch, working directory or\n    config file.\n\n    CAUTION: Locks should only be broken when you are sure that the process\n    holding the lock has been stopped.\n\n    You can get information on what locks are open via the 'brz info\n    [location]' command.\n\n    :Examples:\n        brz break-lock\n        brz break-lock brz+ssh://example.com/brz/foo\n        brz break-lock --conf ~/.config/breezy\n    "
    takes_args = ['location?']
    takes_options = [Option('config', help='LOCATION is the directory where the config lock is.'), Option('force', help='Do not ask for confirmation before breaking the lock.')]

    def run(self, location=None, config=False, force=False):
        if location is None:
            location = '.'
        if force:
            ui.ui_factory = ui.ConfirmationUserInterfacePolicy(ui.ui_factory, None, {'breezy.lockdir.break': True})
        if config:
            conf = _mod_config.LockableConfig(file_name=location)
            conf.break_lock()
        else:
            control, relpath = controldir.ControlDir.open_containing(location)
            try:
                control.break_lock()
            except NotImplementedError:
                pass