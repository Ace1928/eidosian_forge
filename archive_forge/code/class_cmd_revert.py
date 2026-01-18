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
class cmd_revert(Command):
    __doc__ = '    Set files in the working tree back to the contents of a previous revision.\n\n    Giving a list of files will revert only those files.  Otherwise, all files\n    will be reverted.  If the revision is not specified with \'--revision\', the\n    working tree basis revision is used. A revert operation affects only the\n    working tree, not any revision history like the branch and repository or\n    the working tree basis revision.\n\n    To remove only some changes, without reverting to a prior version, use\n    merge instead.  For example, "merge . -r -2..-3" (don\'t forget the ".")\n    will remove the changes introduced by the second last commit (-2), without\n    affecting the changes introduced by the last commit (-1).  To remove\n    certain changes on a hunk-by-hunk basis, see the shelve command.\n    To update the branch to a specific revision or the latest revision and\n    update the working tree accordingly while preserving local changes, see the\n    update command.\n\n    Uncommitted changes to files that are reverted will be discarded.\n    However, by default, any files that have been manually changed will be\n    backed up first.  (Files changed only by merge are not backed up.)  Backup\n    files have \'.~#~\' appended to their name, where # is a number.\n\n    When you provide files, you can use their current pathname or the pathname\n    from the target revision.  So you can use revert to "undelete" a file by\n    name.  If you name a directory, all the contents of that directory will be\n    reverted.\n\n    If you have newly added files since the target revision, they will be\n    removed.  If the files to be removed have been changed, backups will be\n    created as above.  Directories containing unknown files will not be\n    deleted.\n\n    The working tree contains a list of revisions that have been merged but\n    not yet committed. These revisions will be included as additional parents\n    of the next commit.  Normally, using revert clears that list as well as\n    reverting the files.  If any files are specified, revert leaves the list\n    of uncommitted merges alone and reverts only the files.  Use ``brz revert\n    .`` in the tree root to revert all files but keep the recorded merges,\n    and ``brz revert --forget-merges`` to clear the pending merge list without\n    reverting any files.\n\n    Using "brz revert --forget-merges", it is possible to apply all of the\n    changes from a branch in a single revision.  To do this, perform the merge\n    as desired.  Then doing revert with the "--forget-merges" option will keep\n    the content of the tree as it was, but it will clear the list of pending\n    merges.  The next commit will then contain all of the changes that are\n    present in the other branch, but without any other parent revisions.\n    Because this technique forgets where these changes originated, it may\n    cause additional conflicts on later merges involving the same source and\n    target branches.\n    '
    _see_also = ['cat', 'export', 'merge', 'shelve']
    takes_options = ['revision', Option('no-backup', 'Do not save backups of reverted files.'), Option('forget-merges', 'Remove pending merge marker, without changing any files.')]
    takes_args = ['file*']

    def run(self, revision=None, no_backup=False, file_list=None, forget_merges=None):
        tree, file_list = WorkingTree.open_containing_paths(file_list)
        self.enter_context(tree.lock_tree_write())
        if forget_merges:
            tree.set_parent_ids(tree.get_parent_ids()[:1])
        else:
            self._revert_tree_to_revision(tree, revision, file_list, no_backup)

    @staticmethod
    def _revert_tree_to_revision(tree, revision, file_list, no_backup):
        rev_tree = _get_one_revision_tree('revert', revision, tree=tree)
        tree.revert(file_list, rev_tree, not no_backup, None, report_changes=True)