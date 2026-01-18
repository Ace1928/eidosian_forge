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
class cmd_remove_branch(Command):
    __doc__ = 'Remove a branch.\n\n    This will remove the branch from the specified location but\n    will keep any working tree or repository in place.\n\n    :Examples:\n\n      Remove the branch at repo/trunk::\n\n        brz remove-branch repo/trunk\n\n    '
    takes_args = ['location?']
    takes_options = ['directory', Option('force', help='Remove branch even if it is the active branch.')]
    aliases = ['rmbranch']

    def run(self, directory=None, location=None, force=False):
        br = open_nearby_branch(near=directory, location=location)
        if not force and br.controldir.has_workingtree():
            try:
                active_branch = br.controldir.open_branch(name='')
            except errors.NotBranchError as exc:
                active_branch = None
            if active_branch is not None and br.control_url == active_branch.control_url:
                raise errors.CommandError(gettext('Branch is active. Use --force to remove it.'))
        br.controldir.destroy_branch(br.name)