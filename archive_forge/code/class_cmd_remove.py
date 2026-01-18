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
class cmd_remove(Command):
    __doc__ = 'Remove files or directories.\n\n    This makes Breezy stop tracking changes to the specified files. Breezy will\n    delete them if they can easily be recovered using revert otherwise they\n    will be backed up (adding an extension of the form .~#~). If no options or\n    parameters are given Breezy will scan for files that are being tracked by\n    Breezy but missing in your tree and stop tracking them for you.\n    '
    takes_args = ['file*']
    takes_options = ['verbose', Option('new', help='Only remove files that have never been committed.'), RegistryOption.from_kwargs('file-deletion-strategy', 'The file deletion mode to be used.', title='Deletion Strategy', value_switches=True, enum_switch=False, safe='Backup changed files (default).', keep='Delete from brz but leave the working copy.', no_backup="Don't backup changed files.")]
    aliases = ['rm', 'del']
    encoding_type = 'replace'

    def run(self, file_list, verbose=False, new=False, file_deletion_strategy='safe'):
        tree, file_list = WorkingTree.open_containing_paths(file_list)
        if file_list is not None:
            file_list = [f for f in file_list]
        self.enter_context(tree.lock_write())
        if new:
            added = tree.changes_from(tree.basis_tree(), specific_files=file_list).added
            file_list = sorted([f.path[1] for f in added], reverse=True)
            if len(file_list) == 0:
                raise errors.CommandError(gettext('No matching files.'))
        elif file_list is None:
            missing = []
            for change in tree.iter_changes(tree.basis_tree()):
                if change.path[1] is not None and change.kind[1] is None:
                    missing.append(change.path[1])
            file_list = sorted(missing, reverse=True)
            file_deletion_strategy = 'keep'
        tree.remove(file_list, verbose=verbose, to_file=self.outf, keep_files=file_deletion_strategy == 'keep', force=file_deletion_strategy == 'no-backup')