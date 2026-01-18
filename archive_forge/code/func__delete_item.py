from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _delete_item(self, path, inv):
    newly_added = self._new_file_ids.get(path)
    if newly_added:
        file_id = newly_added
        ie = self._delta_entries_by_fileid[file_id][3]
    else:
        file_id = inv.path2id(path)
        if file_id is None:
            self.mutter('ignoring delete of %s as not in inventory', path)
            return
        try:
            ie = inv.get_entry(file_id)
        except errors.NoSuchId:
            self.mutter('ignoring delete of %s as not in inventory', path)
            return
    self.record_delete(path, ie)