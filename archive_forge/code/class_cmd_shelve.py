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
class cmd_shelve(Command):
    __doc__ = 'Temporarily set aside some changes from the current tree.\n\n    Shelve allows you to temporarily put changes you\'ve made "on the shelf",\n    ie. out of the way, until a later time when you can bring them back from\n    the shelf with the \'unshelve\' command.  The changes are stored alongside\n    your working tree, and so they aren\'t propagated along with your branch nor\n    will they survive its deletion.\n\n    If shelve --list is specified, previously-shelved changes are listed.\n\n    Shelve is intended to help separate several sets of changes that have\n    been inappropriately mingled.  If you just want to get rid of all changes\n    and you don\'t need to restore them later, use revert.  If you want to\n    shelve all text changes at once, use shelve --all.\n\n    If filenames are specified, only the changes to those files will be\n    shelved. Other files will be left untouched.\n\n    If a revision is specified, changes since that revision will be shelved.\n\n    You can put multiple items on the shelf, and by default, \'unshelve\' will\n    restore the most recently shelved changes.\n\n    For complicated changes, it is possible to edit the changes in a separate\n    editor program to decide what the file remaining in the working copy\n    should look like.  To do this, add the configuration option\n\n        change_editor = PROGRAM {new_path} {old_path}\n\n    where {new_path} is replaced with the path of the new version of the\n    file and {old_path} is replaced with the path of the old version of\n    the file.  The PROGRAM should save the new file with the desired\n    contents of the file in the working tree.\n\n    '
    takes_args = ['file*']
    takes_options = ['directory', 'revision', Option('all', help='Shelve all changes.'), 'message', RegistryOption('writer', 'Method to use for writing diffs.', breezy.option.diff_writer_registry, value_switches=True, enum_switch=False), Option('list', help='List shelved changes.'), Option('destroy', help='Destroy removed changes instead of shelving them.')]
    _see_also = ['unshelve', 'configuration']

    def run(self, revision=None, all=False, file_list=None, message=None, writer=None, list=False, destroy=False, directory=None):
        if list:
            return self.run_for_list(directory=directory)
        from .shelf_ui import Shelver
        if writer is None:
            writer = breezy.option.diff_writer_registry.get()
        try:
            shelver = Shelver.from_args(writer(self.outf), revision, all, file_list, message, destroy=destroy, directory=directory)
            try:
                shelver.run()
            finally:
                shelver.finalize()
        except errors.UserAbort:
            return 0

    def run_for_list(self, directory=None):
        if directory is None:
            directory = '.'
        tree = WorkingTree.open_containing(directory)[0]
        self.enter_context(tree.lock_read())
        manager = tree.get_shelf_manager()
        shelves = manager.active_shelves()
        if len(shelves) == 0:
            note(gettext('No shelved changes.'))
            return 0
        for shelf_id in reversed(shelves):
            message = manager.get_metadata(shelf_id).get(b'message')
            if message is None:
                message = '<no message>'
            self.outf.write('%3d: %s\n' % (shelf_id, message))
        return 1