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
def _do_interactive(self, merger):
    """Perform an interactive merge.

        This works by generating a preview tree of the merge, then using
        Shelver to selectively remove the differences between the working tree
        and the preview tree.
        """
    from . import shelf_ui
    result_tree = self._get_preview(merger)
    writer = breezy.option.diff_writer_registry.get()
    shelver = shelf_ui.Shelver(merger.this_tree, result_tree, destroy=True, reporter=shelf_ui.ApplyReporter(), diff_writer=writer(self.outf))
    try:
        shelver.run()
    finally:
        shelver.finalize()