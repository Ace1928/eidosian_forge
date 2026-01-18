from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _add_entry(self, entry):
    old_path = entry[0]
    new_path = entry[1]
    file_id = entry[2]
    ie = entry[3]
    existing = self._delta_entries_by_fileid.get(file_id, None)
    if existing is not None:
        old_path = existing[0]
        entry = (old_path, new_path, file_id, ie)
    if new_path is None and old_path is None:
        del self._delta_entries_by_fileid[file_id]
        parent_dir = osutils.dirname(existing[1])
        self.mutter('cancelling add of %s with parent %s' % (existing[1], parent_dir))
        if parent_dir:
            self._dirs_that_might_become_empty.add(parent_dir)
        return
    else:
        self._delta_entries_by_fileid[file_id] = entry
    if new_path is None:
        parent_dir = osutils.dirname(old_path)
        if parent_dir:
            self._dirs_that_might_become_empty.add(parent_dir)
    elif old_path is not None and old_path != new_path:
        old_parent_dir = osutils.dirname(old_path)
        new_parent_dir = osutils.dirname(new_path)
        if old_parent_dir and old_parent_dir != new_parent_dir:
            self._dirs_that_might_become_empty.add(old_parent_dir)
    if file_id in self.per_file_parents_for_commit:
        return
    if old_path is None:
        per_file_parents, ie.revision = self.rev_store.get_parents_and_revision_for_entry(ie)
        self.per_file_parents_for_commit[file_id] = per_file_parents
    elif new_path is None:
        pass
    elif old_path != new_path:
        per_file_parents, _ = self.rev_store.get_parents_and_revision_for_entry(ie)
        self.per_file_parents_for_commit[file_id] = per_file_parents
    else:
        per_file_parents, ie.revision = self.rev_store.get_parents_and_revision_for_entry(ie)
        self.per_file_parents_for_commit[file_id] = per_file_parents