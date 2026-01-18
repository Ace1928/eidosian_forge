import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
def _generate_index_changes(self):
    """Generate an inventory delta for the current transform."""
    removed_id = set(self._removed_id)
    removed_id.update(self._removed_contents)
    changes = {}
    changed_ids = set()
    for id_set in [self._new_name, self._new_parent, self._new_executability, self._new_contents]:
        changed_ids.update(id_set)
    for id_set in [self._new_name, self._new_parent]:
        removed_id.update(id_set)
    changed_kind = set(self._new_contents)
    changed_kind.difference_update(changed_ids)
    changed_kind = (t for t in changed_kind if self.tree_kind(t) != self.final_kind(t))
    changed_ids.update(changed_kind)
    for t in changed_kind:
        if self.final_kind(t) == 'directory':
            removed_id.add(t)
            changed_ids.remove(t)
    new_paths = sorted(FinalPaths(self).get_paths(changed_ids))
    total_entries = len(new_paths) + len(removed_id)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        for num, trans_id in enumerate(removed_id):
            if num % 10 == 0:
                child_pb.update(gettext('removing file'), num, total_entries)
            try:
                path = self._tree_id_paths[trans_id]
            except KeyError:
                continue
            changes[path] = (None, None, None, None)
        for num, (path, trans_id) in enumerate(new_paths):
            if num % 10 == 0:
                child_pb.update(gettext('adding file'), num + len(removed_id), total_entries)
            kind = self.final_kind(trans_id)
            if kind is None:
                continue
            versioned = self.final_is_versioned(trans_id)
            if not versioned:
                continue
            executability = self._new_executability.get(trans_id)
            reference_revision = self._new_reference_revision.get(trans_id)
            symlink_target = self._symlink_target.get(trans_id)
            changes[path] = (kind, executability, reference_revision, symlink_target)
    return [(p, k, e, rr, st) for p, (k, e, rr, st) in changes.items()]