from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class CommonInventory:
    """Basic inventory logic, defined in terms of primitives like has_id.

    An inventory is the metadata about the contents of a tree.

    This is broadly a map from file_id to entries such as directories, files,
    symlinks and tree references. Each entry maintains its own metadata like
    SHA1 and length for files, or children for a directory.

    Entries can be looked up either by path or by file_id.

    InventoryEntry objects must not be modified after they are
    inserted, other than through the Inventory API.
    """

    def has_filename(self, filename):
        return bool(self.path2id(filename))

    def id2path(self, file_id):
        """Return as a string the path to file_id.

        >>> i = Inventory()
        >>> e = i.add(InventoryDirectory(b'src-id', 'src', ROOT_ID))
        >>> e = i.add(InventoryFile(b'foo-id', 'foo.c', parent_id=b'src-id'))
        >>> print(i.id2path(b'foo-id'))
        src/foo.c

        :raises NoSuchId: If file_id is not present in the inventory.
        """
        return '/'.join(reversed([parent.name for parent in self._iter_file_id_parents(file_id)][:-1]))

    def iter_entries(self, from_dir=None, recursive=True):
        """Return (path, entry) pairs, in order by name.

        :param from_dir: if None, start from the root,
          otherwise start from this directory (either file-id or entry)
        :param recursive: recurse into directories or not
        """
        if from_dir is None:
            if self.root is None:
                return
            from_dir = self.root
            yield ('', self.root)
        elif isinstance(from_dir, bytes):
            from_dir = self.get_entry(from_dir)
        children = sorted(from_dir.children.items())
        if not recursive:
            yield from children
            return
        children = deque(children)
        stack = [('', children)]
        while stack:
            from_dir_relpath, children = stack[-1]
            while children:
                name, ie = children.popleft()
                path = from_dir_relpath + '/' + name
                yield (path[1:], ie)
                if ie.kind != 'directory':
                    continue
                new_children = sorted(ie.children.items())
                new_children = deque(new_children)
                stack.append((path, new_children))
                break
            else:
                stack.pop()

    def _preload_cache(self):
        """Populate any caches, we are about to access all items.

        The default implementation does nothing, because CommonInventory doesn't
        have a cache.
        """
        pass

    def iter_entries_by_dir(self, from_dir=None, specific_file_ids=None):
        """Iterate over the entries in a directory first order.

        This returns all entries for a directory before returning
        the entries for children of a directory. This is not
        lexicographically sorted order, and is a hybrid between
        depth-first and breadth-first.

        :return: This yields (path, entry) pairs
        """
        if specific_file_ids and (not isinstance(specific_file_ids, set)):
            specific_file_ids = set(specific_file_ids)
        if from_dir is None and specific_file_ids is None:
            self._preload_cache()
        if from_dir is None:
            if self.root is None:
                return
            if specific_file_ids is not None and len(specific_file_ids) == 1:
                file_id = list(specific_file_ids)[0]
                if file_id is not None:
                    try:
                        path = self.id2path(file_id)
                    except errors.NoSuchId:
                        pass
                    else:
                        yield (path, self.get_entry(file_id))
                return
            from_dir = self.root
            if specific_file_ids is None or self.root.file_id in specific_file_ids:
                yield ('', self.root)
        elif isinstance(from_dir, bytes):
            from_dir = self.get_entry(from_dir)
        else:
            raise TypeError(from_dir)
        if specific_file_ids is not None:
            parents = set()
            byid = self

            def add_ancestors(file_id):
                if not byid.has_id(file_id):
                    return
                parent_id = byid.get_entry(file_id).parent_id
                if parent_id is None:
                    return
                if parent_id not in parents:
                    parents.add(parent_id)
                    add_ancestors(parent_id)
            for file_id in specific_file_ids:
                add_ancestors(file_id)
        else:
            parents = None
        stack = [('', from_dir)]
        while stack:
            cur_relpath, cur_dir = stack.pop()
            child_dirs = []
            for child_name, child_ie in sorted(cur_dir.children.items()):
                child_relpath = cur_relpath + child_name
                if specific_file_ids is None or child_ie.file_id in specific_file_ids:
                    yield (child_relpath, child_ie)
                if child_ie.kind == 'directory':
                    if parents is None or child_ie.file_id in parents:
                        child_dirs.append((child_relpath + '/', child_ie))
            stack.extend(reversed(child_dirs))

    def _make_delta(self, old):
        """Make an inventory delta from two inventories."""
        old_ids = set(old.iter_all_ids())
        new_ids = set(self.iter_all_ids())
        adds = new_ids - old_ids
        deletes = old_ids - new_ids
        common = old_ids.intersection(new_ids)
        delta = []
        for file_id in deletes:
            delta.append((old.id2path(file_id), None, file_id, None))
        for file_id in adds:
            delta.append((None, self.id2path(file_id), file_id, self.get_entry(file_id)))
        for file_id in common:
            if old.get_entry(file_id) != self.get_entry(file_id):
                delta.append((old.id2path(file_id), self.id2path(file_id), file_id, self.get_entry(file_id)))
        return delta

    def make_entry(self, kind, name, parent_id, file_id=None):
        """Simple thunk to breezy.bzr.inventory.make_entry."""
        return make_entry(kind, name, parent_id, file_id)

    def entries(self):
        """Return list of (path, ie) for all entries except the root.

        This may be faster than iter_entries.
        """
        accum = []

        def descend(dir_ie, dir_path):
            kids = sorted(dir_ie.children.items())
            for name, ie in kids:
                child_path = osutils.pathjoin(dir_path, name)
                accum.append((child_path, ie))
                if ie.kind == 'directory':
                    descend(ie, child_path)
        if self.root is not None:
            descend(self.root, '')
        return accum

    def get_entry_by_path_partial(self, relpath):
        """Like get_entry_by_path, but return TreeReference objects.

        :param relpath: Path to resolve, either as string with / as separators,
            or as list of elements.
        :return: tuple with ie, resolved elements and elements left to resolve
        """
        if isinstance(relpath, str):
            names = osutils.splitpath(relpath)
        else:
            names = relpath
        try:
            parent = self.root
        except errors.NoSuchId:
            return (None, None, None)
        if parent is None:
            return (None, None, None)
        for i, f in enumerate(names):
            try:
                children = getattr(parent, 'children', None)
                if children is None:
                    return (None, None, None)
                cie = children[f]
                if cie.kind == 'tree-reference':
                    return (cie, names[:i + 1], names[i + 1:])
                parent = cie
            except KeyError:
                return (None, None, None)
        return (parent, names, [])

    def get_entry_by_path(self, relpath):
        """Return an inventory entry by path.

        :param relpath: may be either a list of path components, or a single
            string, in which case it is automatically split.

        This returns the entry of the last component in the path,
        which may be either a file or a directory.

        Returns None IFF the path is not found.
        """
        if isinstance(relpath, str):
            names = osutils.splitpath(relpath)
        else:
            names = relpath
        try:
            parent = self.root
        except errors.NoSuchId:
            return None
        if parent is None:
            return None
        for f in names:
            try:
                children = getattr(parent, 'children', None)
                if children is None:
                    return None
                cie = children[f]
                parent = cie
            except KeyError:
                return None
        return parent

    def path2id(self, relpath):
        """Walk down through directories to return entry of last component.

        :param relpath: may be either a list of path components, or a single
            string, in which case it is automatically split.

        This returns the entry of the last component in the path,
        which may be either a file or a directory.

        Returns None IFF the path is not found.
        """
        ie = self.get_entry_by_path(relpath)
        if ie is None:
            return None
        return ie.file_id

    def filter(self, specific_fileids):
        """Get an inventory view filtered against a set of file-ids.

        Children of directories and parents are included.

        The result may or may not reference the underlying inventory
        so it should be treated as immutable.
        """
        interesting_parents = set()
        for fileid in specific_fileids:
            try:
                interesting_parents.update(self.get_idpath(fileid))
            except errors.NoSuchId:
                pass
        entries = self.iter_entries()
        if self.root is None:
            return Inventory(root_id=None)
        other = Inventory(next(entries)[1].file_id)
        other.root.revision = self.root.revision
        other.revision_id = self.revision_id
        directories_to_expand = set()
        for path, entry in entries:
            file_id = entry.file_id
            if file_id in specific_fileids or entry.parent_id in directories_to_expand:
                if entry.kind == 'directory':
                    directories_to_expand.add(file_id)
            elif file_id not in interesting_parents:
                continue
            other.add(entry.copy())
        return other

    def get_idpath(self, file_id):
        """Return a list of file_ids for the path to an entry.

        The list contains one element for each directory followed by
        the id of the file itself.  So the length of the returned list
        is equal to the depth of the file in the tree, counting the
        root directory as depth 1.
        """
        p = []
        for parent in self._iter_file_id_parents(file_id):
            p.insert(0, parent.file_id)
        return p