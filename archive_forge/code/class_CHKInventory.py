from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class CHKInventory(CommonInventory):
    """An inventory persisted in a CHK store.

    By design, a CHKInventory is immutable so many of the methods
    supported by Inventory - add, rename, apply_delta, etc - are *not*
    supported. To create a new CHKInventory, use create_by_apply_delta()
    or from_inventory(), say.

    Internally, a CHKInventory has one or two CHKMaps:

    * id_to_entry - a map from (file_id,) => InventoryEntry as bytes
    * parent_id_basename_to_file_id - a map from (parent_id, basename_utf8)
        => file_id as bytes

    The second map is optional and not present in early CHkRepository's.

    No caching is performed: every method call or item access will perform
    requests to the storage layer. As such, keep references to objects you
    want to reuse.
    """

    def __init__(self, search_key_name):
        CommonInventory.__init__(self)
        self._fileid_to_entry_cache = {}
        self._fully_cached = False
        self._path_to_fileid_cache = {}
        self._search_key_name = search_key_name
        self.root_id = None

    def __eq__(self, other):
        """Compare two sets by comparing their contents."""
        if not isinstance(other, CHKInventory):
            return NotImplemented
        this_key = self.id_to_entry.key()
        other_key = other.id_to_entry.key()
        this_pid_key = self.parent_id_basename_to_file_id.key()
        other_pid_key = other.parent_id_basename_to_file_id.key()
        if None in (this_key, this_pid_key, other_key, other_pid_key):
            return False
        return this_key == other_key and this_pid_key == other_pid_key

    def _entry_to_bytes(self, entry):
        """Serialise entry as a single bytestring.

        :param Entry: An inventory entry.
        :return: A bytestring for the entry.

        The BNF:
        ENTRY ::= FILE | DIR | SYMLINK | TREE
        FILE ::= "file: " COMMON SEP SHA SEP SIZE SEP EXECUTABLE
        DIR ::= "dir: " COMMON
        SYMLINK ::= "symlink: " COMMON SEP TARGET_UTF8
        TREE ::= "tree: " COMMON REFERENCE_REVISION
        COMMON ::= FILE_ID SEP PARENT_ID SEP NAME_UTF8 SEP REVISION
        SEP ::= "
"
        """
        if entry.parent_id is not None:
            parent_str = entry.parent_id
        else:
            parent_str = b''
        name_str = entry.name.encode('utf8')
        if entry.kind == 'file':
            if entry.executable:
                exec_str = b'Y'
            else:
                exec_str = b'N'
            return b'file: %s\n%s\n%s\n%s\n%s\n%d\n%s' % (entry.file_id, parent_str, name_str, entry.revision, entry.text_sha1, entry.text_size, exec_str)
        elif entry.kind == 'directory':
            return b'dir: %s\n%s\n%s\n%s' % (entry.file_id, parent_str, name_str, entry.revision)
        elif entry.kind == 'symlink':
            return b'symlink: %s\n%s\n%s\n%s\n%s' % (entry.file_id, parent_str, name_str, entry.revision, entry.symlink_target.encode('utf8'))
        elif entry.kind == 'tree-reference':
            return b'tree: %s\n%s\n%s\n%s\n%s' % (entry.file_id, parent_str, name_str, entry.revision, entry.reference_revision)
        else:
            raise ValueError('unknown kind %r' % entry.kind)

    def _expand_fileids_to_parents_and_children(self, file_ids):
        """Give a more wholistic view starting with the given file_ids.

        For any file_id which maps to a directory, we will include all children
        of that directory. We will also include all directories which are
        parents of the given file_ids, but we will not include their children.

        eg:
          /     # TREE_ROOT
          foo/  # foo-id
            baz # baz-id
            frob/ # frob-id
              fringle # fringle-id
          bar/  # bar-id
            bing # bing-id

        if given [foo-id] we will include
            TREE_ROOT as interesting parents
        and
            foo-id, baz-id, frob-id, fringle-id
        As interesting ids.
        """
        interesting = set()
        directories_to_expand = set()
        children_of_parent_id = {}
        for entry in self._getitems(file_ids):
            if entry.kind == 'directory':
                directories_to_expand.add(entry.file_id)
            interesting.add(entry.parent_id)
            children_of_parent_id.setdefault(entry.parent_id, set()).add(entry.file_id)
        remaining_parents = interesting.difference(file_ids)
        interesting.add(None)
        remaining_parents.discard(None)
        while remaining_parents:
            next_parents = set()
            for entry in self._getitems(remaining_parents):
                next_parents.add(entry.parent_id)
                children_of_parent_id.setdefault(entry.parent_id, set()).add(entry.file_id)
            remaining_parents = next_parents.difference(interesting)
            interesting.update(remaining_parents)
        interesting.update(file_ids)
        interesting.discard(None)
        while directories_to_expand:
            keys = [StaticTuple(f).intern() for f in directories_to_expand]
            directories_to_expand = set()
            items = self.parent_id_basename_to_file_id.iteritems(keys)
            next_file_ids = {item[1] for item in items}
            next_file_ids = next_file_ids.difference(interesting)
            interesting.update(next_file_ids)
            for entry in self._getitems(next_file_ids):
                if entry.kind == 'directory':
                    directories_to_expand.add(entry.file_id)
                children_of_parent_id.setdefault(entry.parent_id, set()).add(entry.file_id)
        return (interesting, children_of_parent_id)

    def filter(self, specific_fileids):
        """Get an inventory view filtered against a set of file-ids.

        Children of directories and parents are included.

        The result may or may not reference the underlying inventory
        so it should be treated as immutable.
        """
        interesting, parent_to_children = self._expand_fileids_to_parents_and_children(specific_fileids)
        other = Inventory(self.root_id)
        other.root.revision = self.root.revision
        other.revision_id = self.revision_id
        if not interesting or not parent_to_children:
            return other
        cache = self._fileid_to_entry_cache
        remaining_children = deque(parent_to_children[self.root_id])
        while remaining_children:
            file_id = remaining_children.popleft()
            ie = cache[file_id]
            if ie.kind == 'directory':
                ie = ie.copy()
            other.add(ie)
            if file_id in parent_to_children:
                remaining_children.extend(parent_to_children[file_id])
        return other

    @staticmethod
    def _bytes_to_utf8name_key(data):
        """Get the file_id, revision_id key out of data."""
        sections = data.split(b'\n')
        kind, file_id = sections[0].split(b': ')
        return (sections[2], file_id, sections[3])

    def _bytes_to_entry(self, bytes):
        """Deserialise a serialised entry."""
        sections = bytes.split(b'\n')
        if sections[0].startswith(b'file: '):
            result = InventoryFile(sections[0][6:], sections[2].decode('utf8'), sections[1])
            result.text_sha1 = sections[4]
            result.text_size = int(sections[5])
            result.executable = sections[6] == b'Y'
        elif sections[0].startswith(b'dir: '):
            result = CHKInventoryDirectory(sections[0][5:], sections[2].decode('utf8'), sections[1], self)
        elif sections[0].startswith(b'symlink: '):
            result = InventoryLink(sections[0][9:], sections[2].decode('utf8'), sections[1])
            result.symlink_target = sections[4].decode('utf8')
        elif sections[0].startswith(b'tree: '):
            result = TreeReference(sections[0][6:], sections[2].decode('utf8'), sections[1])
            result.reference_revision = sections[4]
        else:
            raise ValueError('Not a serialised entry %r' % bytes)
        result.file_id = result.file_id
        result.revision = sections[3]
        if result.parent_id == b'':
            result.parent_id = None
        self._fileid_to_entry_cache[result.file_id] = result
        return result

    def create_by_apply_delta(self, inventory_delta, new_revision_id, propagate_caches=False):
        """Create a new CHKInventory by applying inventory_delta to this one.

        See the inventory developers documentation for the theory behind
        inventory deltas.

        :param inventory_delta: The inventory delta to apply. See
            Inventory.apply_delta for details.
        :param new_revision_id: The revision id of the resulting CHKInventory.
        :param propagate_caches: If True, the caches for this inventory are
          copied to and updated for the result.
        :return: The new CHKInventory.
        """
        split = osutils.split
        result = CHKInventory(self._search_key_name)
        if propagate_caches:
            result._path_to_fileid_cache = self._path_to_fileid_cache.copy()
        search_key_func = chk_map.search_key_registry.get(self._search_key_name)
        self.id_to_entry._ensure_root()
        maximum_size = self.id_to_entry._root_node.maximum_size
        result.revision_id = new_revision_id
        result.id_to_entry = chk_map.CHKMap(self.id_to_entry._store, self.id_to_entry.key(), search_key_func=search_key_func)
        result.id_to_entry._ensure_root()
        result.id_to_entry._root_node.set_maximum_size(maximum_size)
        parent_id_basename_delta = {}
        if self.parent_id_basename_to_file_id is not None:
            result.parent_id_basename_to_file_id = chk_map.CHKMap(self.parent_id_basename_to_file_id._store, self.parent_id_basename_to_file_id.key(), search_key_func=search_key_func)
            result.parent_id_basename_to_file_id._ensure_root()
            self.parent_id_basename_to_file_id._ensure_root()
            result_p_id_root = result.parent_id_basename_to_file_id._root_node
            p_id_root = self.parent_id_basename_to_file_id._root_node
            result_p_id_root.set_maximum_size(p_id_root.maximum_size)
            result_p_id_root._key_width = p_id_root._key_width
        else:
            result.parent_id_basename_to_file_id = None
        result.root_id = self.root_id
        id_to_entry_delta = []
        inventory_delta = _check_delta_unique_ids(inventory_delta)
        inventory_delta = _check_delta_unique_old_paths(inventory_delta)
        inventory_delta = _check_delta_unique_new_paths(inventory_delta)
        inventory_delta = _check_delta_ids_match_entry(inventory_delta)
        inventory_delta = _check_delta_ids_are_valid(inventory_delta)
        inventory_delta = _check_delta_new_path_entry_both_or_None(inventory_delta)
        parents = set()
        deletes = set()
        altered = set()
        for old_path, new_path, file_id, entry in inventory_delta:
            if new_path == '':
                result.root_id = file_id
            if new_path is None:
                new_key = None
                new_value = None
                if propagate_caches:
                    try:
                        del result._path_to_fileid_cache[old_path]
                    except KeyError:
                        pass
                deletes.add(file_id)
            else:
                new_key = StaticTuple(file_id)
                new_value = result._entry_to_bytes(entry)
                result._path_to_fileid_cache[new_path] = file_id
                parents.add((split(new_path)[0], entry.parent_id))
            if old_path is None:
                old_key = None
            else:
                old_key = StaticTuple(file_id)
                if self.id2path(file_id) != old_path:
                    raise errors.InconsistentDelta(old_path, file_id, 'Entry was at wrong other path %r.' % self.id2path(file_id))
                altered.add(file_id)
            id_to_entry_delta.append(StaticTuple(old_key, new_key, new_value))
            if result.parent_id_basename_to_file_id is not None:
                if old_path is None:
                    old_key = None
                else:
                    old_entry = self.get_entry(file_id)
                    old_key = self._parent_id_basename_key(old_entry)
                if new_path is None:
                    new_key = None
                    new_value = None
                else:
                    new_key = self._parent_id_basename_key(entry)
                    new_value = file_id
                if old_key != new_key:
                    if old_key is not None:
                        parent_id_basename_delta.setdefault(old_key, [None, None])[0] = old_key
                    if new_key is not None:
                        parent_id_basename_delta.setdefault(new_key, [None, None])[1] = new_value
        for file_id in deletes:
            entry = self.get_entry(file_id)
            if entry.kind != 'directory':
                continue
            for child in entry.children.values():
                if child.file_id not in altered:
                    raise errors.InconsistentDelta(self.id2path(child.file_id), child.file_id, 'Child not deleted or reparented when parent deleted.')
        result.id_to_entry.apply_delta(id_to_entry_delta)
        if parent_id_basename_delta:
            delta_list = []
            for key, (old_key, value) in parent_id_basename_delta.items():
                if value is not None:
                    delta_list.append((old_key, key, value))
                else:
                    delta_list.append((old_key, None, None))
            result.parent_id_basename_to_file_id.apply_delta(delta_list)
        parents.discard(('', None))
        for parent_path, parent in parents:
            try:
                if result.get_entry(parent).kind != 'directory':
                    raise errors.InconsistentDelta(result.id2path(parent), parent, 'Not a directory, but given children')
            except errors.NoSuchId:
                raise errors.InconsistentDelta('<unknown>', parent, 'Parent is not present in resulting inventory.')
            if result.path2id(parent_path) != parent:
                raise errors.InconsistentDelta(parent_path, parent, 'Parent has wrong path %r.' % result.path2id(parent_path))
        return result

    @classmethod
    def deserialise(klass, chk_store, lines, expected_revision_id):
        """Deserialise a CHKInventory.

        :param chk_store: A CHK capable VersionedFiles instance.
        :param bytes: The serialised bytes.
        :param expected_revision_id: The revision ID we think this inventory is
            for.
        :return: A CHKInventory
        """
        if not lines[-1].endswith(b'\n'):
            raise ValueError('last line should have trailing eol\n')
        if lines[0] != b'chkinventory:\n':
            raise ValueError('not a serialised CHKInventory: %r' % bytes)
        info = {}
        allowed_keys = frozenset((b'root_id', b'revision_id', b'parent_id_basename_to_file_id', b'search_key_name', b'id_to_entry'))
        for line in lines[1:]:
            key, value = line.rstrip(b'\n').split(b': ', 1)
            if key not in allowed_keys:
                raise errors.BzrError('Unknown key in inventory: %r\n%r' % (key, bytes))
            if key in info:
                raise errors.BzrError('Duplicate key in inventory: %r\n%r' % (key, bytes))
            info[key] = value
        revision_id = info[b'revision_id']
        root_id = info[b'root_id']
        search_key_name = info.get(b'search_key_name', b'plain')
        parent_id_basename_to_file_id = info.get(b'parent_id_basename_to_file_id', None)
        if not parent_id_basename_to_file_id.startswith(b'sha1:'):
            raise ValueError('parent_id_basename_to_file_id should be a sha1 key not %r' % (parent_id_basename_to_file_id,))
        id_to_entry = info[b'id_to_entry']
        if not id_to_entry.startswith(b'sha1:'):
            raise ValueError('id_to_entry should be a sha1 key not %r' % (id_to_entry,))
        result = CHKInventory(search_key_name)
        result.revision_id = revision_id
        result.root_id = root_id
        search_key_func = chk_map.search_key_registry.get(result._search_key_name)
        if parent_id_basename_to_file_id is not None:
            result.parent_id_basename_to_file_id = chk_map.CHKMap(chk_store, StaticTuple(parent_id_basename_to_file_id), search_key_func=search_key_func)
        else:
            result.parent_id_basename_to_file_id = None
        result.id_to_entry = chk_map.CHKMap(chk_store, StaticTuple(id_to_entry), search_key_func=search_key_func)
        if (result.revision_id,) != expected_revision_id:
            raise ValueError('Mismatched revision id and expected: %r, %r' % (result.revision_id, expected_revision_id))
        return result

    @classmethod
    def from_inventory(klass, chk_store, inventory, maximum_size=0, search_key_name=b'plain'):
        """Create a CHKInventory from an existing inventory.

        The content of inventory is copied into the chk_store, and a
        CHKInventory referencing that is returned.

        :param chk_store: A CHK capable VersionedFiles instance.
        :param inventory: The inventory to copy.
        :param maximum_size: The CHKMap node size limit.
        :param search_key_name: The identifier for the search key function
        """
        result = klass(search_key_name)
        result.revision_id = inventory.revision_id
        result.root_id = inventory.root.file_id
        entry_to_bytes = result._entry_to_bytes
        parent_id_basename_key = result._parent_id_basename_key
        id_to_entry_dict = {}
        parent_id_basename_dict = {}
        for path, entry in inventory.iter_entries():
            key = StaticTuple(entry.file_id).intern()
            id_to_entry_dict[key] = entry_to_bytes(entry)
            p_id_key = parent_id_basename_key(entry)
            parent_id_basename_dict[p_id_key] = entry.file_id
        result._populate_from_dicts(chk_store, id_to_entry_dict, parent_id_basename_dict, maximum_size=maximum_size)
        return result

    def _populate_from_dicts(self, chk_store, id_to_entry_dict, parent_id_basename_dict, maximum_size):
        search_key_func = chk_map.search_key_registry.get(self._search_key_name)
        root_key = chk_map.CHKMap.from_dict(chk_store, id_to_entry_dict, maximum_size=maximum_size, key_width=1, search_key_func=search_key_func)
        self.id_to_entry = chk_map.CHKMap(chk_store, root_key, search_key_func)
        root_key = chk_map.CHKMap.from_dict(chk_store, parent_id_basename_dict, maximum_size=maximum_size, key_width=2, search_key_func=search_key_func)
        self.parent_id_basename_to_file_id = chk_map.CHKMap(chk_store, root_key, search_key_func)

    def _parent_id_basename_key(self, entry):
        """Create a key for a entry in a parent_id_basename_to_file_id index."""
        if entry.parent_id is not None:
            parent_id = entry.parent_id
        else:
            parent_id = b''
        return StaticTuple(parent_id, entry.name.encode('utf8')).intern()

    def get_entry(self, file_id):
        """map a single file_id -> InventoryEntry."""
        if file_id is None:
            raise errors.NoSuchId(self, file_id)
        result = self._fileid_to_entry_cache.get(file_id, None)
        if result is not None:
            return result
        try:
            return self._bytes_to_entry(next(self.id_to_entry.iteritems([StaticTuple(file_id)]))[1])
        except StopIteration:
            raise errors.NoSuchId(self, file_id)

    def _getitems(self, file_ids):
        """Similar to get_entry, but lets you query for multiple.

        The returned order is undefined. And currently if an item doesn't
        exist, it isn't included in the output.
        """
        result = []
        remaining = []
        for file_id in file_ids:
            entry = self._fileid_to_entry_cache.get(file_id, None)
            if entry is None:
                remaining.append(file_id)
            else:
                result.append(entry)
        file_keys = [StaticTuple(f).intern() for f in remaining]
        for file_key, value in self.id_to_entry.iteritems(file_keys):
            entry = self._bytes_to_entry(value)
            result.append(entry)
            self._fileid_to_entry_cache[entry.file_id] = entry
        return result

    def has_id(self, file_id):
        if self._fileid_to_entry_cache.get(file_id, None) is not None:
            return True
        return len(list(self.id_to_entry.iteritems([StaticTuple(file_id)]))) == 1

    def is_root(self, file_id):
        return file_id == self.root_id

    def _iter_file_id_parents(self, file_id):
        """Yield the parents of file_id up to the root."""
        while file_id is not None:
            try:
                ie = self.get_entry(file_id)
            except KeyError:
                raise errors.NoSuchId(tree=self, file_id=file_id)
            yield ie
            file_id = ie.parent_id

    def iter_all_ids(self):
        """Iterate over all file-ids."""
        for key, _ in self.id_to_entry.iteritems():
            yield key[-1]

    def iter_just_entries(self):
        """Iterate over all entries.

        Unlike iter_entries(), just the entries are returned (not (path, ie))
        and the order of entries is undefined.

        XXX: We may not want to merge this into bzr.dev.
        """
        for key, entry in self.id_to_entry.iteritems():
            file_id = key[0]
            ie = self._fileid_to_entry_cache.get(file_id, None)
            if ie is None:
                ie = self._bytes_to_entry(entry)
                self._fileid_to_entry_cache[file_id] = ie
            yield ie

    def _preload_cache(self):
        """Make sure all file-ids are in _fileid_to_entry_cache"""
        if self._fully_cached:
            return
        cache = self._fileid_to_entry_cache
        for key, entry in self.id_to_entry.iteritems():
            file_id = key[0]
            if file_id not in cache:
                ie = self._bytes_to_entry(entry)
                cache[file_id] = ie
            else:
                ie = cache[file_id]
        last_parent_id = last_parent_ie = None
        pid_items = self.parent_id_basename_to_file_id.iteritems()
        for key, child_file_id in pid_items:
            if key == (b'', b''):
                if child_file_id != self.root_id:
                    raise ValueError('Data inconsistency detected. We expected data with key ("","") to match the root id, but %s != %s' % (child_file_id, self.root_id))
                continue
            parent_id, basename = key
            ie = cache[child_file_id]
            if parent_id == last_parent_id:
                parent_ie = last_parent_ie
            else:
                parent_ie = cache[parent_id]
            if parent_ie.kind != 'directory':
                raise ValueError('Data inconsistency detected. An entry in the parent_id_basename_to_file_id map has parent_id {%s} but the kind of that object is %r not "directory"' % (parent_id, parent_ie.kind))
            if parent_ie._children is None:
                parent_ie._children = {}
            basename = basename.decode('utf-8')
            if basename in parent_ie._children:
                existing_ie = parent_ie._children[basename]
                if existing_ie != ie:
                    raise ValueError('Data inconsistency detected. Two entries with basename %r were found in the parent entry {%s}' % (basename, parent_id))
            if basename != ie.name:
                raise ValueError('Data inconsistency detected. In the parent_id_basename_to_file_id map, file_id {%s} is listed as having basename %r, but in the id_to_entry map it is %r' % (child_file_id, basename, ie.name))
            parent_ie._children[basename] = ie
        self._fully_cached = True

    def iter_changes(self, basis):
        """Generate a Tree.iter_changes change list between this and basis.

        :param basis: Another CHKInventory.
        :return: An iterator over the changes between self and basis, as per
            tree.iter_changes().
        """
        for key, basis_value, self_value in self.id_to_entry.iter_changes(basis.id_to_entry):
            file_id = key[0]
            if basis_value is not None:
                basis_entry = basis._bytes_to_entry(basis_value)
                path_in_source = basis.id2path(file_id)
                basis_parent = basis_entry.parent_id
                basis_name = basis_entry.name
                basis_executable = basis_entry.executable
            else:
                path_in_source = None
                basis_parent = None
                basis_name = None
                basis_executable = None
            if self_value is not None:
                self_entry = self._bytes_to_entry(self_value)
                path_in_target = self.id2path(file_id)
                self_parent = self_entry.parent_id
                self_name = self_entry.name
                self_executable = self_entry.executable
            else:
                path_in_target = None
                self_parent = None
                self_name = None
                self_executable = None
            if basis_value is None:
                kind = (None, self_entry.kind)
                versioned = (False, True)
            elif self_value is None:
                kind = (basis_entry.kind, None)
                versioned = (True, False)
            else:
                kind = (basis_entry.kind, self_entry.kind)
                versioned = (True, True)
            changed_content = False
            if kind[0] != kind[1]:
                changed_content = True
            elif kind[0] == 'file':
                if self_entry.text_size != basis_entry.text_size or self_entry.text_sha1 != basis_entry.text_sha1:
                    changed_content = True
            elif kind[0] == 'symlink':
                if self_entry.symlink_target != basis_entry.symlink_target:
                    changed_content = True
            elif kind[0] == 'tree-reference':
                if self_entry.reference_revision != basis_entry.reference_revision:
                    changed_content = True
            parent = (basis_parent, self_parent)
            name = (basis_name, self_name)
            executable = (basis_executable, self_executable)
            if not changed_content and parent[0] == parent[1] and (name[0] == name[1]) and (executable[0] == executable[1]):
                continue
            yield (file_id, (path_in_source, path_in_target), changed_content, versioned, parent, name, kind, executable)

    def __len__(self):
        """Return the number of entries in the inventory."""
        return len(self.id_to_entry)

    def _make_delta(self, old):
        """Make an inventory delta from two inventories."""
        if not isinstance(old, CHKInventory):
            return CommonInventory._make_delta(self, old)
        delta = []
        for key, old_value, self_value in self.id_to_entry.iter_changes(old.id_to_entry):
            file_id = key[0]
            if old_value is not None:
                old_path = old.id2path(file_id)
            else:
                old_path = None
            if self_value is not None:
                entry = self._bytes_to_entry(self_value)
                self._fileid_to_entry_cache[file_id] = entry
                new_path = self.id2path(file_id)
            else:
                entry = None
                new_path = None
            delta.append((old_path, new_path, file_id, entry))
        return delta

    def path2id(self, relpath):
        """See CommonInventory.path2id()."""
        if isinstance(relpath, str):
            names = osutils.splitpath(relpath)
        else:
            names = relpath
            if relpath == []:
                relpath = ['']
            relpath = osutils.pathjoin(*relpath)
        result = self._path_to_fileid_cache.get(relpath, None)
        if result is not None:
            return result
        current_id = self.root_id
        if current_id is None:
            return None
        parent_id_index = self.parent_id_basename_to_file_id
        cur_path = None
        for basename in names:
            if cur_path is None:
                cur_path = basename
            else:
                cur_path = cur_path + '/' + basename
            basename_utf8 = basename.encode('utf8')
            file_id = self._path_to_fileid_cache.get(cur_path, None)
            if file_id is None:
                key_filter = [StaticTuple(current_id, basename_utf8)]
                items = parent_id_index.iteritems(key_filter)
                for (parent_id, name_utf8), file_id in items:
                    if parent_id != current_id or name_utf8 != basename_utf8:
                        raise errors.BzrError('corrupt inventory lookup! %r %r %r %r' % (parent_id, current_id, name_utf8, basename_utf8))
                if file_id is None:
                    return None
                else:
                    self._path_to_fileid_cache[cur_path] = file_id
            current_id = file_id
        return current_id

    def to_lines(self):
        """Serialise the inventory to lines."""
        lines = [b'chkinventory:\n']
        if self._search_key_name != b'plain':
            lines.append(b'search_key_name: %s\n' % self._search_key_name)
            lines.append(b'root_id: %s\n' % self.root_id)
            lines.append(b'parent_id_basename_to_file_id: %s\n' % (self.parent_id_basename_to_file_id.key()[0],))
            lines.append(b'revision_id: %s\n' % self.revision_id)
            lines.append(b'id_to_entry: %s\n' % (self.id_to_entry.key()[0],))
        else:
            lines.append(b'revision_id: %s\n' % self.revision_id)
            lines.append(b'root_id: %s\n' % self.root_id)
            if self.parent_id_basename_to_file_id is not None:
                lines.append(b'parent_id_basename_to_file_id: %s\n' % (self.parent_id_basename_to_file_id.key()[0],))
            lines.append(b'id_to_entry: %s\n' % (self.id_to_entry.key()[0],))
        return lines

    @property
    def root(self):
        """Get the root entry."""
        return self.get_entry(self.root_id)