import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class CombinedGraphIndex:
    """A GraphIndex made up from smaller GraphIndices.

    The backing indices must implement GraphIndex, and are presumed to be
    static data.

    Queries against the combined index will be made against the first index,
    and then the second and so on. The order of indices can thus influence
    performance significantly. For example, if one index is on local disk and a
    second on a remote server, the local disk index should be before the other
    in the index list.

    Also, queries tend to need results from the same indices as previous
    queries.  So the indices will be reordered after every query to put the
    indices that had the result(s) of that query first (while otherwise
    preserving the relative ordering).
    """

    def __init__(self, indices, reload_func=None):
        """Create a CombinedGraphIndex backed by indices.

        :param indices: An ordered list of indices to query for data.
        :param reload_func: A function to call if we find we are missing an
            index. Should have the form reload_func() => True/False to indicate
            if reloading actually changed anything.
        """
        self._indices = indices
        self._reload_func = reload_func
        self._sibling_indices = []
        self._index_names = [None] * len(self._indices)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(map(repr, self._indices)))

    def clear_cache(self):
        """See GraphIndex.clear_cache()"""
        for index in self._indices:
            index.clear_cache()

    def get_parent_map(self, keys):
        """See graph.StackedParentsProvider.get_parent_map"""
        search_keys = set(keys)
        if _mod_revision.NULL_REVISION in search_keys:
            search_keys.discard(_mod_revision.NULL_REVISION)
            found_parents = {_mod_revision.NULL_REVISION: []}
        else:
            found_parents = {}
        for index, key, value, refs in self.iter_entries(search_keys):
            parents = refs[0]
            if not parents:
                parents = (_mod_revision.NULL_REVISION,)
            found_parents[key] = parents
        return found_parents
    __contains__ = _has_key_from_parent_map

    def insert_index(self, pos, index, name=None):
        """Insert a new index in the list of indices to query.

        :param pos: The position to insert the index.
        :param index: The index to insert.
        :param name: a name for this index, e.g. a pack name.  These names can
            be used to reflect index reorderings to related CombinedGraphIndex
            instances that use the same names.  (see set_sibling_indices)
        """
        self._indices.insert(pos, index)
        self._index_names.insert(pos, name)

    def iter_all_entries(self):
        """Iterate over all keys within the index

        Duplicate keys across child indices are presumed to have the same
        value and are only reported once.

        :return: An iterable of (index, key, reference_lists, value).
            There is no defined order for the result iteration - it will be in
            the most efficient order for the index.
        """
        seen_keys = set()
        while True:
            try:
                for index in self._indices:
                    for node in index.iter_all_entries():
                        if node[1] not in seen_keys:
                            yield node
                            seen_keys.add(node[1])
                return
            except _mod_transport.NoSuchFile as e:
                if not self._try_reload(e):
                    raise

    def iter_entries(self, keys):
        """Iterate over keys within the index.

        Duplicate keys across child indices are presumed to have the same
        value and are only reported once.

        :param keys: An iterable providing the keys to be retrieved.
        :return: An iterable of (index, key, reference_lists, value). There is
            no defined order for the result iteration - it will be in the most
            efficient order for the index.
        """
        keys = set(keys)
        hit_indices = []
        while True:
            try:
                for index in self._indices:
                    if not keys:
                        break
                    index_hit = False
                    for node in index.iter_entries(keys):
                        keys.remove(node[1])
                        yield node
                        index_hit = True
                    if index_hit:
                        hit_indices.append(index)
                break
            except _mod_transport.NoSuchFile as e:
                if not self._try_reload(e):
                    raise
        self._move_to_front(hit_indices)

    def iter_entries_prefix(self, keys):
        """Iterate over keys within the index using prefix matching.

        Duplicate keys across child indices are presumed to have the same
        value and are only reported once.

        Prefix matching is applied within the tuple of a key, not to within
        the bytestring of each key element. e.g. if you have the keys ('foo',
        'bar'), ('foobar', 'gam') and do a prefix search for ('foo', None) then
        only the former key is returned.

        :param keys: An iterable providing the key prefixes to be retrieved.
            Each key prefix takes the form of a tuple the length of a key, but
            with the last N elements 'None' rather than a regular bytestring.
            The first element cannot be 'None'.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys with a matching prefix to those supplied. No additional keys
            will be returned, and every match that is in the index will be
            returned.
        """
        keys = set(keys)
        if not keys:
            return
        seen_keys = set()
        hit_indices = []
        while True:
            try:
                for index in self._indices:
                    index_hit = False
                    for node in index.iter_entries_prefix(keys):
                        if node[1] in seen_keys:
                            continue
                        seen_keys.add(node[1])
                        yield node
                        index_hit = True
                    if index_hit:
                        hit_indices.append(index)
                break
            except _mod_transport.NoSuchFile as e:
                if not self._try_reload(e):
                    raise
        self._move_to_front(hit_indices)

    def _move_to_front(self, hit_indices):
        """Rearrange self._indices so that hit_indices are first.

        Order is maintained as much as possible, e.g. the first unhit index
        will be the first index in _indices after the hit_indices, and the
        hit_indices will be present in exactly the order they are passed to
        _move_to_front.

        _move_to_front propagates to all objects in self._sibling_indices by
        calling _move_to_front_by_name.
        """
        if self._indices[:len(hit_indices)] == hit_indices:
            return
        hit_names = self._move_to_front_by_index(hit_indices)
        for sibling_idx in self._sibling_indices:
            sibling_idx._move_to_front_by_name(hit_names)

    def _move_to_front_by_index(self, hit_indices):
        """Core logic for _move_to_front.

        Returns a list of names corresponding to the hit_indices param.
        """
        indices_info = zip(self._index_names, self._indices)
        if 'index' in debug.debug_flags:
            indices_info = list(indices_info)
            trace.mutter('CombinedGraphIndex reordering: currently %r, promoting %r', indices_info, hit_indices)
        hit_names = []
        unhit_names = []
        new_hit_indices = []
        unhit_indices = []
        for offset, (name, idx) in enumerate(indices_info):
            if idx in hit_indices:
                hit_names.append(name)
                new_hit_indices.append(idx)
                if len(new_hit_indices) == len(hit_indices):
                    unhit_names.extend(self._index_names[offset + 1:])
                    unhit_indices.extend(self._indices[offset + 1:])
                    break
            else:
                unhit_names.append(name)
                unhit_indices.append(idx)
        self._indices = new_hit_indices + unhit_indices
        self._index_names = hit_names + unhit_names
        if 'index' in debug.debug_flags:
            trace.mutter('CombinedGraphIndex reordered: %r', self._indices)
        return hit_names

    def _move_to_front_by_name(self, hit_names):
        """Moves indices named by 'hit_names' to front of the search order, as
        described in _move_to_front.
        """
        indices_info = zip(self._index_names, self._indices)
        hit_indices = []
        for name, idx in indices_info:
            if name in hit_names:
                hit_indices.append(idx)
        self._move_to_front_by_index(hit_indices)

    def find_ancestry(self, keys, ref_list_num):
        """Find the complete ancestry for the given set of keys.

        Note that this is a whole-ancestry request, so it should be used
        sparingly.

        :param keys: An iterable of keys to look for
        :param ref_list_num: The reference list which references the parents
            we care about.
        :return: (parent_map, missing_keys)
        """
        missing_keys = set()
        parent_map = {}
        keys_to_lookup = set(keys)
        generation = 0
        while keys_to_lookup:
            generation += 1
            all_index_missing = None
            for index_idx, index in enumerate(self._indices):
                index_missing_keys = set()
                search_keys = keys_to_lookup
                sub_generation = 0
                while search_keys:
                    sub_generation += 1
                    search_keys = index._find_ancestors(search_keys, ref_list_num, parent_map, index_missing_keys)
                keys_to_lookup = index_missing_keys
                if all_index_missing is None:
                    all_index_missing = set(index_missing_keys)
                else:
                    all_index_missing.intersection_update(index_missing_keys)
                if not keys_to_lookup:
                    break
            if all_index_missing is None:
                missing_keys.update(keys_to_lookup)
                keys_to_lookup = None
            else:
                missing_keys.update(all_index_missing)
                keys_to_lookup.difference_update(all_index_missing)
        return (parent_map, missing_keys)

    def key_count(self):
        """Return an estimate of the number of keys in this index.

        For CombinedGraphIndex this is approximated by the sum of the keys of
        the child indices. As child indices may have duplicate keys this can
        have a maximum error of the number of child indices * largest number of
        keys in any index.
        """
        while True:
            try:
                return sum((index.key_count() for index in self._indices), 0)
            except _mod_transport.NoSuchFile as e:
                if not self._try_reload(e):
                    raise
    missing_keys = _missing_keys_from_parent_map

    def _try_reload(self, error):
        """We just got a NoSuchFile exception.

        Try to reload the indices, if it fails, just raise the current
        exception.
        """
        if self._reload_func is None:
            return False
        trace.mutter('Trying to reload after getting exception: %s', str(error))
        if not self._reload_func():
            trace.mutter('_reload_func indicated nothing has changed. Raising original exception.')
            return False
        return True

    def set_sibling_indices(self, sibling_combined_graph_indices):
        """Set the CombinedGraphIndex objects to reorder after reordering self.
        """
        self._sibling_indices = sibling_combined_graph_indices

    def validate(self):
        """Validate that everything in the index can be accessed."""
        while True:
            try:
                for index in self._indices:
                    index.validate()
                return
            except _mod_transport.NoSuchFile as e:
                if not self._try_reload(e):
                    raise