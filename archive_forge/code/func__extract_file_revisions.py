from contextlib import ExitStack
import time
from typing import Type
from breezy import registry
from breezy import revision as _mod_revision
from breezy.osutils import format_date, local_time_offset
def _extract_file_revisions(self):
    """Extract the working revisions for all files"""
    if self._tree is None:
        return
    self._clean = True
    with ExitStack() as es:
        if self._working_tree is self._tree:
            basis_tree = self._working_tree.basis_tree()
            es.enter_context(self._working_tree.lock_read())
        else:
            basis_tree = self._branch.repository.revision_tree(self._revision_id)
        es.enter_context(basis_tree.lock_read())
        for info in basis_tree.list_files(include_root=True):
            self._file_revisions[info[0]] = info[-1].revision
        if not self._check or self._working_tree is not self._tree:
            return
        delta = self._working_tree.changes_from(basis_tree, include_root=True, want_unversioned=True)
        for change in delta.renamed:
            self._clean = False
            self._file_revisions[change.path[0]] = 'renamed to {}'.format(change.path[1])
        for change in delta.removed:
            self._clean = False
            self._file_revisions[change.path[0]] = 'removed'
        for change in delta.added:
            self._clean = False
            self._file_revisions[change.path[1]] = 'new'
        for change in delta.renamed:
            self._clean = False
            self._file_revisions[change.path[1]] = 'renamed from {}'.format(change.path[0])
        for change in delta.copied:
            self._clean = False
            self._file_revisions[change.path[1]] = 'copied from {}'.format(change.path[0])
        for change in delta.modified:
            self._clean = False
            self._file_revisions[change.path[1]] = 'modified'
        for change in delta.unversioned:
            self._clean = False
            self._file_revisions[change.path[1]] = 'unversioned'