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
class cmd_info(Command):
    __doc__ = 'Show information about a working tree, branch or repository.\n\n    This command will show all known locations and formats associated to the\n    tree, branch or repository.\n\n    In verbose mode, statistical information is included with each report.\n    To see extended statistic information, use a verbosity level of 2 or\n    higher by specifying the verbose option multiple times, e.g. -vv.\n\n    Branches and working trees will also report any missing revisions.\n\n    :Examples:\n\n      Display information on the format and related locations:\n\n        brz info\n\n      Display the above together with extended format information and\n      basic statistics (like the number of files in the working tree and\n      number of revisions in the branch and repository):\n\n        brz info -v\n\n      Display the above together with number of committers to the branch:\n\n        brz info -vv\n    '
    _see_also = ['revno', 'working-trees', 'repositories']
    takes_args = ['location?']
    takes_options = ['verbose']
    encoding_type = 'replace'

    @display_command
    def run(self, location=None, verbose=False):
        if verbose:
            noise_level = get_verbosity_level()
        else:
            noise_level = 0
        from .info import show_bzrdir_info
        show_bzrdir_info(controldir.ControlDir.open_containing(location)[0], verbose=noise_level, outfile=self.outf)