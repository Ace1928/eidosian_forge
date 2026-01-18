from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _ensure_directory(self, path, inv):
    """Ensure that the containing directory exists for 'path'"""
    dirname, basename = osutils.split(path)
    if dirname == '':
        return (basename, self.inventory_root_id)
    try:
        ie = self._get_directory_entry(inv, dirname)
    except KeyError:
        pass
    else:
        return (basename, ie.file_id)
    dir_basename, parent_id = self._ensure_directory(dirname, inv)
    dir_file_id = self.bzr_file_id(dirname)
    ie = inventory.entry_factory['directory'](dir_file_id, dir_basename, parent_id)
    ie.revision = self.revision_id
    self.directory_entries[dirname] = ie
    self.data_for_commit[dir_file_id] = b''
    if inv.has_id(dir_file_id):
        self.record_delete(dirname, ie)
    self.record_new(dirname, ie)
    return (basename, ie.file_id)