from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _update_branches(self, old_revno, old_revid, new_revno):
    """Update the master and local branch to the new revision.

        This will try to make sure that the master branch is updated
        before the local branch.

        :param old_revno: Revision number of master branch before the
            commit
        :param old_revid: Tip of master branch before the commit
        :param new_revno: Revision number of the new commit
        """
    if not self.builder.updates_branch:
        self._process_pre_hooks(old_revno, new_revno)
        if self.bound_branch:
            self._set_progress_stage('Uploading data to master branch')
            new_revno, self.rev_id = self.master_branch.import_last_revision_info_and_tags(self.branch, new_revno, self.rev_id, lossy=self._lossy)
            if self._lossy:
                self.branch.fetch(self.master_branch, self.rev_id)
        if new_revno is None:
            new_revno = 1
        self.branch.set_last_revision_info(new_revno, self.rev_id)
    else:
        try:
            self._process_pre_hooks(old_revno, new_revno)
        except BaseException:
            self.branch.set_last_revision_info(old_revno, old_revid)
            raise
    if self.bound_branch:
        self._set_progress_stage('Merging tags to master branch')
        tag_updates, tag_conflicts = self.branch.tags.merge_to(self.master_branch.tags)
        if tag_conflicts:
            warning_lines = ['    ' + name for name, _, _ in tag_conflicts]
            note(gettext('Conflicting tags in bound branch:\n{}'.format('\n'.join(warning_lines))))