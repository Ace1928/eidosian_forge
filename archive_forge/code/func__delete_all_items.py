from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _delete_all_items(self, inv):
    if len(inv) == 0:
        return
    for path, ie in inv.iter_entries_by_dir():
        if path != '':
            self.record_delete(path, ie)