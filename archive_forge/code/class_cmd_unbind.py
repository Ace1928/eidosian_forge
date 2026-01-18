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
class cmd_unbind(Command):
    __doc__ = 'Convert the current checkout into a regular branch.\n\n    After unbinding, the local branch is considered independent and subsequent\n    commits will be local only.\n    '
    _see_also = ['checkouts', 'bind']
    takes_options = ['directory']

    def run(self, directory='.'):
        b, relpath = Branch.open_containing(directory)
        if not b.unbind():
            raise errors.CommandError(gettext('Local branch is not bound'))