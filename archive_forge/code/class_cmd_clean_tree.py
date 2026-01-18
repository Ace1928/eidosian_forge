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
class cmd_clean_tree(Command):
    __doc__ = "Remove unwanted files from working tree.\n\n    By default, only unknown files, not ignored files, are deleted.  Versioned\n    files are never deleted.\n\n    Another class is 'detritus', which includes files emitted by brz during\n    normal operations and selftests.  (The value of these files decreases with\n    time.)\n\n    If no options are specified, unknown files are deleted.  Otherwise, option\n    flags are respected, and may be combined.\n\n    To check what clean-tree will do, use --dry-run.\n    "
    takes_options = ['directory', Option('ignored', help='Delete all ignored files.'), Option('detritus', help='Delete conflict files, merge and revert backups, and failed selftest dirs.'), Option('unknown', help='Delete files unknown to brz (default).'), Option('dry-run', help='Show files to delete instead of deleting them.'), Option('force', help='Do not prompt before deleting.')]

    def run(self, unknown=False, ignored=False, detritus=False, dry_run=False, force=False, directory='.'):
        from .clean_tree import clean_tree
        if not (unknown or ignored or detritus):
            unknown = True
        if dry_run:
            force = True
        clean_tree(directory, unknown=unknown, ignored=ignored, detritus=detritus, dry_run=dry_run, no_prompt=force)