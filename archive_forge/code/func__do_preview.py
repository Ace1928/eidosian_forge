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
def _do_preview(self, merger):
    from .diff import show_diff_trees
    result_tree = self._get_preview(merger)
    path_encoding = osutils.get_diff_header_encoding()
    show_diff_trees(merger.this_tree, result_tree, self.outf, old_label='', new_label='', path_encoding=path_encoding)