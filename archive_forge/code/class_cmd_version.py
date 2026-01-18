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
class cmd_version(Command):
    __doc__ = 'Show version of brz.'
    encoding_type = 'replace'
    takes_options = [Option('short', help='Print just the version number.')]

    @display_command
    def run(self, short=False):
        from .version import show_version
        if short:
            self.outf.write(breezy.version_string + '\n')
        else:
            show_version(to_file=self.outf)