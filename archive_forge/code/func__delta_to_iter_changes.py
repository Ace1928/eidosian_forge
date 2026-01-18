from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def _delta_to_iter_changes(self):
    """Convert the inv_delta into an iter_changes repr."""
    basis_inv = self._basis_inv
    for old_path, new_path, file_id, ie in self._inv_delta:
        try:
            old_ie = basis_inv.get_entry(file_id)
        except errors.NoSuchId:
            old_ie = None
            if ie is None:
                raise AssertionError('How is both old and new None?')
                change = InventoryTreeChange(file_id, (old_path, new_path), False, (False, False), (None, None), (None, None), (None, None), (None, None))
            change = InventoryTreeChange(file_id, (old_path, new_path), True, (False, True), (None, ie.parent_id), (None, ie.name), (None, ie.kind), (None, ie.executable))
        else:
            if ie is None:
                change = InventoryTreeChange(file_id, (old_path, new_path), True, (True, False), (old_ie.parent_id, None), (old_ie.name, None), (old_ie.kind, None), (old_ie.executable, None))
            else:
                content_modified = ie.text_sha1 != old_ie.text_sha1 or ie.text_size != old_ie.text_size
                change = InventoryTreeChange(file_id, (old_path, new_path), content_modified, (True, True), (old_ie.parent_id, ie.parent_id), (old_ie.name, ie.name), (old_ie.kind, ie.kind), (old_ie.executable, ie.executable))
        yield change