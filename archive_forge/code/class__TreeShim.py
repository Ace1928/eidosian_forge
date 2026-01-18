from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
class _TreeShim:
    """Fake a Tree implementation.

    This implements just enough of the tree api to make commit builder happy.
    """

    def __init__(self, repo, basis_inv, inv_delta, content_provider):
        self._repo = repo
        self._content_provider = content_provider
        self._basis_inv = basis_inv
        self._inv_delta = inv_delta
        self._new_info_by_id = {file_id: (new_path, ie) for _, new_path, file_id, ie in inv_delta}
        self._new_info_by_path = {new_path: ie for _, new_path, file_id, ie in inv_delta}

    def id2path(self, file_id, recurse='down'):
        if file_id in self._new_info_by_id:
            new_path = self._new_info_by_id[file_id][0]
            if new_path is None:
                raise errors.NoSuchId(self, file_id)
            return new_path
        return self._basis_inv.id2path(file_id)

    def path2id(self, path):
        try:
            return self._new_info_by_path[path].file_id
        except KeyError:
            return self._basis_inv.path2id(path)

    def get_file_with_stat(self, path):
        content = self.get_file_text(path)
        sio = BytesIO(content)
        return (sio, None)

    def get_file_text(self, path):
        file_id = self.path2id(path)
        try:
            return self._content_provider(file_id)
        except KeyError:
            assert file_id not in self._new_info_by_id
            old_ie = self._basis_inv.get_entry(file_id)
            old_text_key = (file_id, old_ie.revision)
            stream = self._repo.texts.get_record_stream([old_text_key], 'unordered', True)
            return next(stream).get_bytes_as('fulltext')

    def get_symlink_target(self, path):
        try:
            ie = self._new_info_by_path[path]
        except KeyError:
            file_id = self.path2id(path)
            return self._basis_inv.get_entry(file_id).symlink_target
        else:
            return ie.symlink_target

    def get_reference_revision(self, path):
        raise NotImplementedError(_TreeShim.get_reference_revision)

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