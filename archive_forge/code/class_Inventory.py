from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class Inventory(CommonInventory):
    """Mutable dict based in-memory inventory.

    We never store the full path to a file, because renaming a directory
    implicitly moves all of its contents.  This class internally maintains a
    lookup tree that allows the children under a directory to be
    returned quickly.

    >>> inv = Inventory()
    >>> inv.add(InventoryFile(b'123-123', 'hello.c', ROOT_ID))
    InventoryFile(b'123-123', 'hello.c', parent_id=b'TREE_ROOT', sha1=None, len=None, revision=None)
    >>> inv.get_entry(b'123-123').name
    'hello.c'

    Id's may be looked up from paths:

    >>> inv.path2id('hello.c')
    b'123-123'
    >>> inv.has_id(b'123-123')
    True

    There are iterators over the contents:

    >>> [entry[0] for entry in inv.iter_entries()]
    ['', 'hello.c']
    """

    def __init__(self, root_id=ROOT_ID, revision_id=None):
        """Create or read an inventory.

        If a working directory is specified, the inventory is read
        from there.  If the file is specified, read from that. If not,
        the inventory is created empty.

        The inventory is created with a default root directory, with
        an id of None.
        """
        if root_id is not None:
            self._set_root(InventoryDirectory(root_id, '', None))
        else:
            self.root = None
            self._byid = {}
        self.revision_id = revision_id

    def __repr__(self):
        max_len = 2048
        closing = '...}'
        contents = repr(self._byid)
        if len(contents) > max_len:
            contents = contents[:max_len - len(closing)] + closing
        return '<Inventory object at {:x}, contents={!r}>'.format(id(self), contents)

    def apply_delta(self, delta):
        """Apply a delta to this inventory.

        See the inventory developers documentation for the theory behind
        inventory deltas.

        If delta application fails the inventory is left in an indeterminate
        state and must not be used.

        :param delta: A list of changes to apply. After all the changes are
            applied the final inventory must be internally consistent, but it
            is ok to supply changes which, if only half-applied would have an
            invalid result - such as supplying two changes which rename two
            files, 'A' and 'B' with each other : [('A', 'B', b'A-id', a_entry),
            ('B', 'A', b'B-id', b_entry)].

            Each change is a tuple, of the form (old_path, new_path, file_id,
            new_entry).

            When new_path is None, the change indicates the removal of an entry
            from the inventory and new_entry will be ignored (using None is
            appropriate). If new_path is not None, then new_entry must be an
            InventoryEntry instance, which will be incorporated into the
            inventory (and replace any existing entry with the same file id).

            When old_path is None, the change indicates the addition of
            a new entry to the inventory.

            When neither new_path nor old_path are None, the change is a
            modification to an entry, such as a rename, reparent, kind change
            etc.

            The children attribute of new_entry is ignored. This is because
            this method preserves children automatically across alterations to
            the parent of the children, and cases where the parent id of a
            child is changing require the child to be passed in as a separate
            change regardless. E.g. in the recursive deletion of a directory -
            the directory's children must be included in the delta, or the
            final inventory will be invalid.

            Note that a file_id must only appear once within a given delta.
            An AssertionError is raised otherwise.
        """
        list(_check_delta_unique_ids(_check_delta_unique_new_paths(_check_delta_unique_old_paths(_check_delta_ids_match_entry(_check_delta_ids_are_valid(_check_delta_new_path_entry_both_or_None(delta)))))))
        children = {}
        for old_path, file_id in sorted(((op, f) for op, np, f, e in delta if op is not None), reverse=True):
            file_id_children = getattr(self.get_entry(file_id), 'children', {})
            if len(file_id_children):
                children[file_id] = file_id_children
            if self.id2path(file_id) != old_path:
                raise errors.InconsistentDelta(old_path, file_id, 'Entry was at wrong other path %r.' % self.id2path(file_id))
            self.remove_recursive_id(file_id)
        for new_path, f, new_entry in sorted(((np, f, e) for op, np, f, e in delta if np is not None)):
            if new_entry.kind == 'directory':
                replacement = InventoryDirectory(new_entry.file_id, new_entry.name, new_entry.parent_id)
                replacement.revision = new_entry.revision
                replacement.children = children.pop(replacement.file_id, {})
                new_entry = replacement
            try:
                self.add(new_entry)
            except DuplicateFileId:
                raise errors.InconsistentDelta(new_path, new_entry.file_id, 'New id is already present in target.')
            except AttributeError:
                raise errors.InconsistentDelta(new_path, new_entry.file_id, 'Parent is not a directory.')
            if self.id2path(new_entry.file_id) != new_path:
                raise errors.InconsistentDelta(new_path, new_entry.file_id, 'New path is not consistent with parent path.')
        if len(children):
            parent_id, children = children.popitem()
            raise errors.InconsistentDelta('<deleted>', parent_id, 'The file id was deleted but its children were not deleted.')

    def create_by_apply_delta(self, inventory_delta, new_revision_id, propagate_caches=False):
        """See CHKInventory.create_by_apply_delta()"""
        new_inv = self.copy()
        new_inv.apply_delta(inventory_delta)
        new_inv.revision_id = new_revision_id
        return new_inv

    def _set_root(self, ie):
        self.root = ie
        self._byid = {self.root.file_id: self.root}

    def copy(self):
        entries = self.iter_entries()
        if self.root is None:
            return Inventory(root_id=None)
        other = Inventory(next(entries)[1].file_id)
        other.root.revision = self.root.revision
        for path, entry in entries:
            other.add(entry.copy())
        return other

    def iter_all_ids(self):
        """Iterate over all file-ids."""
        return iter(self._byid)

    def iter_just_entries(self):
        """Iterate over all entries.

        Unlike iter_entries(), just the entries are returned (not (path, ie))
        and the order of entries is undefined.

        XXX: We may not want to merge this into bzr.dev.
        """
        if self.root is None:
            return ()
        return self._byid.values()

    def __len__(self):
        """Returns number of entries."""
        return len(self._byid)

    def get_entry(self, file_id):
        """Return the entry for given file_id.

        >>> inv = Inventory()
        >>> inv.add(InventoryFile(b'123123', 'hello.c', ROOT_ID))
        InventoryFile(b'123123', 'hello.c', parent_id=b'TREE_ROOT', sha1=None, len=None, revision=None)
        >>> inv.get_entry(b'123123').name
        'hello.c'
        """
        if not isinstance(file_id, bytes):
            raise TypeError(file_id)
        try:
            return self._byid[file_id]
        except KeyError:
            raise errors.NoSuchId(self, file_id)

    def get_file_kind(self, file_id):
        return self._byid[file_id].kind

    def get_child(self, parent_id, filename):
        return self.get_entry(parent_id).children.get(filename)

    def _add_child(self, entry):
        """Add an entry to the inventory, without adding it to its parent"""
        if entry.file_id in self._byid:
            raise errors.BzrError('inventory already contains entry with id {%s}' % entry.file_id)
        self._byid[entry.file_id] = entry
        children = getattr(entry, 'children', {})
        if children is not None:
            for child in children.values():
                self._add_child(child)
        return entry

    def add(self, entry):
        """Add entry to inventory.

        :return: entry
        """
        if entry.file_id in self._byid:
            raise DuplicateFileId(entry.file_id, self._byid[entry.file_id])
        if entry.parent_id is None:
            self.root = entry
        else:
            try:
                parent = self._byid[entry.parent_id]
            except KeyError:
                raise errors.InconsistentDelta('<unknown>', entry.parent_id, 'Parent not in inventory.')
            if entry.name in parent.children:
                raise errors.InconsistentDelta(self.id2path(parent.children[entry.name].file_id), entry.file_id, 'Path already versioned')
            parent.children[entry.name] = entry
        return self._add_child(entry)

    def add_path(self, relpath, kind, file_id=None, parent_id=None):
        """Add entry from a path.

        The immediate parent must already be versioned.

        Returns the new entry object."""
        parts = osutils.splitpath(relpath)
        if len(parts) == 0:
            if file_id is None:
                file_id = generate_ids.gen_root_id()
            self.root = InventoryDirectory(file_id, '', None)
            self._byid = {self.root.file_id: self.root}
            return self.root
        else:
            parent_path = parts[:-1]
            parent_id = self.path2id(parent_path)
            if parent_id is None:
                raise errors.NotVersionedError(path=parent_path)
        ie = make_entry(kind, parts[-1], parent_id, file_id)
        return self.add(ie)

    def delete(self, file_id):
        """Remove entry by id.

        >>> inv = Inventory()
        >>> inv.add(InventoryFile(b'123', 'foo.c', ROOT_ID))
        InventoryFile(b'123', 'foo.c', parent_id=b'TREE_ROOT', sha1=None, len=None, revision=None)
        >>> inv.has_id(b'123')
        True
        >>> inv.delete(b'123')
        >>> inv.has_id(b'123')
        False
        """
        ie = self.get_entry(file_id)
        del self._byid[file_id]
        if ie.parent_id is not None:
            del self.get_entry(ie.parent_id).children[ie.name]

    def __eq__(self, other):
        """Compare two sets by comparing their contents.

        >>> i1 = Inventory()
        >>> i2 = Inventory()
        >>> i1 == i2
        True
        >>> i1.add(InventoryFile(b'123', 'foo', ROOT_ID))
        InventoryFile(b'123', 'foo', parent_id=b'TREE_ROOT', sha1=None, len=None, revision=None)
        >>> i1 == i2
        False
        >>> i2.add(InventoryFile(b'123', 'foo', ROOT_ID))
        InventoryFile(b'123', 'foo', parent_id=b'TREE_ROOT', sha1=None, len=None, revision=None)
        >>> i1 == i2
        True
        """
        if not isinstance(other, Inventory):
            return NotImplemented
        return self._byid == other._byid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        raise ValueError('not hashable')

    def _iter_file_id_parents(self, file_id):
        """Yield the parents of file_id up to the root."""
        while file_id is not None:
            try:
                ie = self._byid[file_id]
            except KeyError:
                raise errors.NoSuchId(tree=None, file_id=file_id)
            yield ie
            file_id = ie.parent_id

    def has_id(self, file_id):
        return file_id in self._byid

    def _make_delta(self, old):
        """Make an inventory delta from two inventories."""
        old_getter = old.get_entry
        new_getter = self.get_entry
        old_ids = set(old.iter_all_ids())
        new_ids = set(self.iter_all_ids())
        adds = new_ids - old_ids
        deletes = old_ids - new_ids
        if not adds and (not deletes):
            common = new_ids
        else:
            common = old_ids.intersection(new_ids)
        delta = []
        for file_id in deletes:
            delta.append((old.id2path(file_id), None, file_id, None))
        for file_id in adds:
            delta.append((None, self.id2path(file_id), file_id, self.get_entry(file_id)))
        for file_id in common:
            new_ie = new_getter(file_id)
            old_ie = old_getter(file_id)
            if old_ie is new_ie or old_ie == new_ie:
                continue
            else:
                delta.append((old.id2path(file_id), self.id2path(file_id), file_id, new_ie))
        return delta

    def remove_recursive_id(self, file_id):
        """Remove file_id, and children, from the inventory.

        :param file_id: A file_id to remove.
        """
        to_find_delete = [self._byid[file_id]]
        to_delete = []
        while to_find_delete:
            ie = to_find_delete.pop()
            to_delete.append(ie.file_id)
            if ie.kind == 'directory':
                to_find_delete.extend(ie.children.values())
        for file_id in reversed(to_delete):
            ie = self.get_entry(file_id)
            del self._byid[file_id]
        if ie.parent_id is not None:
            del self.get_entry(ie.parent_id).children[ie.name]
        else:
            self.root = None

    def rename(self, file_id, new_parent_id, new_name):
        """Move a file within the inventory.

        This can change either the name, or the parent, or both.

        This does not move the working file.
        """
        new_name = ensure_normalized_name(new_name)
        if not is_valid_name(new_name):
            raise errors.BzrError('not an acceptable filename: %r' % new_name)
        new_parent = self._byid[new_parent_id]
        if new_name in new_parent.children:
            raise errors.BzrError('%r already exists in %r' % (new_name, self.id2path(new_parent_id)))
        new_parent_idpath = self.get_idpath(new_parent_id)
        if file_id in new_parent_idpath:
            raise errors.BzrError('cannot move directory %r into a subdirectory of itself, %r' % (self.id2path(file_id), self.id2path(new_parent_id)))
        file_ie = self._byid[file_id]
        old_parent = self._byid[file_ie.parent_id]
        del old_parent.children[file_ie.name]
        new_parent.children[new_name] = file_ie
        file_ie.name = new_name
        file_ie.parent_id = new_parent_id

    def is_root(self, file_id):
        return self.root is not None and file_id == self.root.file_id