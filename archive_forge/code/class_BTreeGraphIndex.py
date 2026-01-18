from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
class BTreeGraphIndex:
    """Access to nodes via the standard GraphIndex interface for B+Tree's.

    Individual nodes are held in a LRU cache. This holds the root node in
    memory except when very large walks are done.
    """

    def __init__(self, transport, name, size, unlimited_cache=False, offset=0):
        """Create a B+Tree index object on the index name.

        :param transport: The transport to read data for the index from.
        :param name: The file name of the index on transport.
        :param size: Optional size of the index in bytes. This allows
            compatibility with the GraphIndex API, as well as ensuring that
            the initial read (to read the root node header) can be done
            without over-reading even on empty indices, and on small indices
            allows single-IO to read the entire index.
        :param unlimited_cache: If set to True, then instead of using an
            LRUCache with size _NODE_CACHE_SIZE, we will use a dict and always
            cache all leaf nodes.
        :param offset: The start of the btree index data isn't byte 0 of the
            file. Instead it starts at some point later.
        """
        self._transport = transport
        self._name = name
        self._size = size
        self._file = None
        self._recommended_pages = self._compute_recommended_pages()
        self._root_node = None
        self._base_offset = offset
        self._leaf_factory = _LeafNode
        self._leaf_value_cache = None
        if unlimited_cache:
            self._leaf_node_cache = {}
            self._internal_node_cache = {}
        else:
            self._leaf_node_cache = lru_cache.LRUCache(_NODE_CACHE_SIZE)
            self._internal_node_cache = fifo_cache.FIFOCache(100)
        self._key_count = None
        self._row_lengths = None
        self._row_offsets = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        """Equal when self and other were created with the same parameters."""
        return isinstance(self, type(other)) and self._transport == other._transport and (self._name == other._name) and (self._size == other._size)

    def __lt__(self, other):
        if isinstance(other, type(self)):
            return (self._name, self._size) < (other._name, other._size)
        if isinstance(other, BTreeBuilder):
            return True
        raise TypeError

    def __ne__(self, other):
        return not self.__eq__(other)

    def _get_and_cache_nodes(self, nodes):
        """Read nodes and cache them in the lru.

        The nodes list supplied is sorted and then read from disk, each node
        being inserted it into the _node_cache.

        Note: Asking for more nodes than the _node_cache can contain will
        result in some of the results being immediately discarded, to prevent
        this an assertion is raised if more nodes are asked for than are
        cachable.

        :return: A dict of {node_pos: node}
        """
        found = {}
        start_of_leaves = None
        for node_pos, node in self._read_nodes(sorted(nodes)):
            if node_pos == 0:
                self._root_node = node
            else:
                if start_of_leaves is None:
                    start_of_leaves = self._row_offsets[-2]
                if node_pos < start_of_leaves:
                    self._internal_node_cache[node_pos] = node
                else:
                    self._leaf_node_cache[node_pos] = node
            found[node_pos] = node
        return found

    def _compute_recommended_pages(self):
        """Convert transport's recommended_page_size into btree pages.

        recommended_page_size is in bytes, we want to know how many _PAGE_SIZE
        pages fit in that length.
        """
        recommended_read = self._transport.recommended_page_size()
        recommended_pages = int(math.ceil(recommended_read / _PAGE_SIZE))
        return recommended_pages

    def _compute_total_pages_in_index(self):
        """How many pages are in the index.

        If we have read the header we will use the value stored there.
        Otherwise it will be computed based on the length of the index.
        """
        if self._size is None:
            raise AssertionError('_compute_total_pages_in_index should not be called when self._size is None')
        if self._root_node is not None:
            return self._row_offsets[-1]
        total_pages = int(math.ceil(self._size / _PAGE_SIZE))
        return total_pages

    def _expand_offsets(self, offsets):
        """Find extra pages to download.

        The idea is that we always want to make big-enough requests (like 64kB
        for http), so that we don't waste round trips. So given the entries
        that we already have cached and the new pages being downloaded figure
        out what other pages we might want to read.

        See also doc/developers/btree_index_prefetch.txt for more details.

        :param offsets: The offsets to be read
        :return: A list of offsets to download
        """
        if 'index' in debug.debug_flags:
            trace.mutter('expanding: %s\toffsets: %s', self._name, offsets)
        if len(offsets) >= self._recommended_pages:
            if 'index' in debug.debug_flags:
                trace.mutter('  not expanding large request (%s >= %s)', len(offsets), self._recommended_pages)
            return offsets
        if self._size is None:
            if 'index' in debug.debug_flags:
                trace.mutter('  not expanding without knowing index size')
            return offsets
        total_pages = self._compute_total_pages_in_index()
        cached_offsets = self._get_offsets_to_cached_pages()
        if total_pages - len(cached_offsets) <= self._recommended_pages:
            if cached_offsets:
                expanded = [x for x in range(total_pages) if x not in cached_offsets]
            else:
                expanded = list(range(total_pages))
            if 'index' in debug.debug_flags:
                trace.mutter('  reading all unread pages: %s', expanded)
            return expanded
        if self._root_node is None:
            final_offsets = offsets
        else:
            tree_depth = len(self._row_lengths)
            if len(cached_offsets) < tree_depth and len(offsets) == 1:
                if 'index' in debug.debug_flags:
                    trace.mutter('  not expanding on first reads')
                return offsets
            final_offsets = self._expand_to_neighbors(offsets, cached_offsets, total_pages)
        final_offsets = sorted(final_offsets)
        if 'index' in debug.debug_flags:
            trace.mutter('expanded:  %s', final_offsets)
        return final_offsets

    def _expand_to_neighbors(self, offsets, cached_offsets, total_pages):
        """Expand requests to neighbors until we have enough pages.

        This is called from _expand_offsets after policy has determined that we
        want to expand.
        We only want to expand requests within a given layer. We cheat a little
        bit and assume all requests will be in the same layer. This is true
        given the current design, but if it changes this algorithm may perform
        oddly.

        :param offsets: requested offsets
        :param cached_offsets: offsets for pages we currently have cached
        :return: A set() of offsets after expansion
        """
        final_offsets = set(offsets)
        first = end = None
        new_tips = set(final_offsets)
        while len(final_offsets) < self._recommended_pages and new_tips:
            next_tips = set()
            for pos in new_tips:
                if first is None:
                    first, end = self._find_layer_first_and_end(pos)
                previous = pos - 1
                if previous > 0 and previous not in cached_offsets and (previous not in final_offsets) and (previous >= first):
                    next_tips.add(previous)
                after = pos + 1
                if after < total_pages and after not in cached_offsets and (after not in final_offsets) and (after < end):
                    next_tips.add(after)
            final_offsets.update(next_tips)
            new_tips = next_tips
        return final_offsets

    def clear_cache(self):
        """Clear out any cached/memoized values.

        This can be called at any time, but generally it is used when we have
        extracted some information, but don't expect to be requesting any more
        from this index.
        """
        self._leaf_node_cache.clear()

    def external_references(self, ref_list_num):
        if self._root_node is None:
            self._get_root_node()
        if ref_list_num + 1 > self.node_ref_lists:
            raise ValueError('No ref list %d, index has %d ref lists' % (ref_list_num, self.node_ref_lists))
        keys = set()
        refs = set()
        for node in self.iter_all_entries():
            keys.add(node[1])
            refs.update(node[3][ref_list_num])
        return refs - keys

    def _find_layer_first_and_end(self, offset):
        """Find the start/stop nodes for the layer corresponding to offset.

        :return: (first, end)
            first is the first node in this layer
            end is the first node of the next layer
        """
        first = end = 0
        for roffset in self._row_offsets:
            first = end
            end = roffset
            if offset < roffset:
                break
        return (first, end)

    def _get_offsets_to_cached_pages(self):
        """Determine what nodes we already have cached."""
        cached_offsets = set(self._internal_node_cache)
        cached_offsets.update(self._leaf_node_cache.keys())
        if self._root_node is not None:
            cached_offsets.add(0)
        return cached_offsets

    def _get_root_node(self):
        if self._root_node is None:
            self._get_internal_nodes([0])
        return self._root_node

    def _get_nodes(self, cache, node_indexes):
        found = {}
        needed = []
        for idx in node_indexes:
            if idx == 0 and self._root_node is not None:
                found[0] = self._root_node
                continue
            try:
                found[idx] = cache[idx]
            except KeyError:
                needed.append(idx)
        if not needed:
            return found
        needed = self._expand_offsets(needed)
        found.update(self._get_and_cache_nodes(needed))
        return found

    def _get_internal_nodes(self, node_indexes):
        """Get a node, from cache or disk.

        After getting it, the node will be cached.
        """
        return self._get_nodes(self._internal_node_cache, node_indexes)

    def _cache_leaf_values(self, nodes):
        """Cache directly from key => value, skipping the btree."""
        if self._leaf_value_cache is not None:
            for node in nodes.values():
                for key, value in node.all_items():
                    if key in self._leaf_value_cache:
                        break
                    self._leaf_value_cache[key] = value

    def _get_leaf_nodes(self, node_indexes):
        """Get a bunch of nodes, from cache or disk."""
        found = self._get_nodes(self._leaf_node_cache, node_indexes)
        self._cache_leaf_values(found)
        return found

    def iter_all_entries(self):
        """Iterate over all keys within the index.

        :return: An iterable of (index, key, value) or
            (index, key, value, reference_lists).
            The former tuple is used when there are no reference lists in the
            index, making the API compatible with simple key:value index types.
            There is no defined order for the result iteration - it will be in
            the most efficient order for the index.
        """
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(3, 'iter_all_entries scales with size of history.')
        if not self.key_count():
            return
        if self._row_offsets[-1] == 1:
            if self.node_ref_lists:
                for key, (value, refs) in self._root_node.all_items():
                    yield (self, key, value, refs)
            else:
                for key, (value, refs) in self._root_node.all_items():
                    yield (self, key, value)
            return
        start_of_leaves = self._row_offsets[-2]
        end_of_leaves = self._row_offsets[-1]
        needed_offsets = list(range(start_of_leaves, end_of_leaves))
        if needed_offsets == [0]:
            nodes = [(0, self._root_node)]
        else:
            nodes = self._read_nodes(needed_offsets)
        if self.node_ref_lists:
            for _, node in nodes:
                for key, (value, refs) in node.all_items():
                    yield (self, key, value, refs)
        else:
            for _, node in nodes:
                for key, (value, refs) in node.all_items():
                    yield (self, key, value)

    @staticmethod
    def _multi_bisect_right(in_keys, fixed_keys):
        """Find the positions where each 'in_key' would fit in fixed_keys.

        This is equivalent to doing "bisect_right" on each in_key into
        fixed_keys

        :param in_keys: A sorted list of keys to match with fixed_keys
        :param fixed_keys: A sorted list of keys to match against
        :return: A list of (integer position, [key list]) tuples.
        """
        if not in_keys:
            return []
        if not fixed_keys:
            return [(0, in_keys)]
        if len(in_keys) == 1:
            return [(bisect.bisect_right(fixed_keys, in_keys[0]), in_keys)]
        in_keys_iter = iter(in_keys)
        fixed_keys_iter = enumerate(fixed_keys)
        cur_in_key = next(in_keys_iter)
        cur_fixed_offset, cur_fixed_key = next(fixed_keys_iter)

        class InputDone(Exception):
            pass

        class FixedDone(Exception):
            pass
        output = []
        cur_out = []
        try:
            while True:
                if cur_in_key < cur_fixed_key:
                    cur_keys = []
                    cur_out = (cur_fixed_offset, cur_keys)
                    output.append(cur_out)
                    while cur_in_key < cur_fixed_key:
                        cur_keys.append(cur_in_key)
                        try:
                            cur_in_key = next(in_keys_iter)
                        except StopIteration as exc:
                            raise InputDone from exc
                while cur_in_key >= cur_fixed_key:
                    try:
                        cur_fixed_offset, cur_fixed_key = next(fixed_keys_iter)
                    except StopIteration as exc:
                        raise FixedDone from exc
        except InputDone:
            pass
        except FixedDone:
            cur_keys = [cur_in_key]
            cur_keys.extend(in_keys_iter)
            cur_out = (len(fixed_keys), cur_keys)
            output.append(cur_out)
        return output

    def _walk_through_internal_nodes(self, keys):
        """Take the given set of keys, and find the corresponding LeafNodes.

        :param keys: An unsorted iterable of keys to search for
        :return: (nodes, index_and_keys)
            nodes is a dict mapping {index: LeafNode}
            keys_at_index is a list of tuples of [(index, [keys for Leaf])]
        """
        keys_at_index = [(0, sorted(keys))]
        for row_pos, next_row_start in enumerate(self._row_offsets[1:-1]):
            node_indexes = [idx for idx, s_keys in keys_at_index]
            nodes = self._get_internal_nodes(node_indexes)
            next_nodes_and_keys = []
            for node_index, sub_keys in keys_at_index:
                node = nodes[node_index]
                positions = self._multi_bisect_right(sub_keys, node.keys)
                node_offset = next_row_start + node.offset
                next_nodes_and_keys.extend([(node_offset + pos, s_keys) for pos, s_keys in positions])
            keys_at_index = next_nodes_and_keys
        node_indexes = [idx for idx, s_keys in keys_at_index]
        nodes = self._get_leaf_nodes(node_indexes)
        return (nodes, keys_at_index)

    def iter_entries(self, keys):
        """Iterate over keys within the index.

        :param keys: An iterable providing the keys to be retrieved.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys supplied. No additional keys will be returned, and every
            key supplied that is in the index will be returned.
        """
        keys = frozenset(keys)
        if not keys:
            return
        if not self.key_count():
            return
        needed_keys = []
        if self._leaf_value_cache is None:
            needed_keys = keys
        else:
            for key in keys:
                value = self._leaf_value_cache.get(key, None)
                if value is not None:
                    value, refs = value
                    if self.node_ref_lists:
                        yield (self, key, value, refs)
                    else:
                        yield (self, key, value)
                else:
                    needed_keys.append(key)
        needed_keys = keys
        if not needed_keys:
            return
        nodes, nodes_and_keys = self._walk_through_internal_nodes(needed_keys)
        for node_index, sub_keys in nodes_and_keys:
            if not sub_keys:
                continue
            node = nodes[node_index]
            for next_sub_key in sub_keys:
                if next_sub_key in node:
                    value, refs = node[next_sub_key]
                    if self.node_ref_lists:
                        yield (self, next_sub_key, value, refs)
                    else:
                        yield (self, next_sub_key, value)

    def _find_ancestors(self, keys, ref_list_num, parent_map, missing_keys):
        """Find the parent_map information for the set of keys.

        This populates the parent_map dict and missing_keys set based on the
        queried keys. It also can fill out an arbitrary number of parents that
        it finds while searching for the supplied keys.

        It is unlikely that you want to call this directly. See
        "CombinedGraphIndex.find_ancestry()" for a more appropriate API.

        :param keys: A keys whose ancestry we want to return
            Every key will either end up in 'parent_map' or 'missing_keys'.
        :param ref_list_num: This index in the ref_lists is the parents we
            care about.
        :param parent_map: {key: parent_keys} for keys that are present in this
            index. This may contain more entries than were in 'keys', that are
            reachable ancestors of the keys requested.
        :param missing_keys: keys which are known to be missing in this index.
            This may include parents that were not directly requested, but we
            were able to determine that they are not present in this index.
        :return: search_keys    parents that were found but not queried to know
            if they are missing or present. Callers can re-query this index for
            those keys, and they will be placed into parent_map or missing_keys
        """
        if not self.key_count():
            missing_keys.update(keys)
            return set()
        if ref_list_num >= self.node_ref_lists:
            raise ValueError('No ref list %d, index has %d ref lists' % (ref_list_num, self.node_ref_lists))
        nodes, nodes_and_keys = self._walk_through_internal_nodes(keys)
        parents_not_on_page = set()
        for node_index, sub_keys in nodes_and_keys:
            if not sub_keys:
                continue
            node = nodes[node_index]
            parents_to_check = set()
            for next_sub_key in sub_keys:
                if next_sub_key not in node:
                    missing_keys.add(next_sub_key)
                else:
                    value, refs = node[next_sub_key]
                    parent_keys = refs[ref_list_num]
                    parent_map[next_sub_key] = parent_keys
                    parents_to_check.update(parent_keys)
            parents_to_check = parents_to_check.difference(parent_map)
            while parents_to_check:
                next_parents_to_check = set()
                for key in parents_to_check:
                    if key in node:
                        value, refs = node[key]
                        parent_keys = refs[ref_list_num]
                        parent_map[key] = parent_keys
                        next_parents_to_check.update(parent_keys)
                    elif key < node.min_key:
                        parents_not_on_page.add(key)
                    elif key > node.max_key:
                        parents_not_on_page.add(key)
                    else:
                        missing_keys.add(key)
                parents_to_check = next_parents_to_check.difference(parent_map)
        search_keys = parents_not_on_page.difference(parent_map).difference(missing_keys)
        return search_keys

    def iter_entries_prefix(self, keys):
        """Iterate over keys within the index using prefix matching.

        Prefix matching is applied within the tuple of a key, not to within
        the bytestring of each key element. e.g. if you have the keys ('foo',
        'bar'), ('foobar', 'gam') and do a prefix search for ('foo', None) then
        only the former key is returned.

        WARNING: Note that this method currently causes a full index parse
        unconditionally (which is reasonably appropriate as it is a means for
        thunking many small indices into one larger one and still supplies
        iter_all_entries at the thunk layer).

        :param keys: An iterable providing the key prefixes to be retrieved.
            Each key prefix takes the form of a tuple the length of a key, but
            with the last N elements 'None' rather than a regular bytestring.
            The first element cannot be 'None'.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys with a matching prefix to those supplied. No additional keys
            will be returned, and every match that is in the index will be
            returned.
        """
        keys = sorted(set(keys))
        if not keys:
            return
        if self._key_count is None:
            self._get_root_node()
        nodes = {}
        if self.node_ref_lists:
            if self._key_length == 1:
                for _1, key, value, refs in self.iter_all_entries():
                    nodes[key] = (value, refs)
            else:
                nodes_by_key = {}
                for _1, key, value, refs in self.iter_all_entries():
                    key_value = (key, value, refs)
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = key_value
        elif self._key_length == 1:
            for _1, key, value in self.iter_all_entries():
                nodes[key] = value
        else:
            nodes_by_key = {}
            for _1, key, value in self.iter_all_entries():
                key_value = (key, value)
                key_dict = nodes_by_key
                for subkey in key[:-1]:
                    key_dict = key_dict.setdefault(subkey, {})
                key_dict[key[-1]] = key_value
        if self._key_length == 1:
            for key in keys:
                index._sanity_check_key(self, key)
                try:
                    if self.node_ref_lists:
                        value, node_refs = nodes[key]
                        yield (self, key, value, node_refs)
                    else:
                        yield (self, key, nodes[key])
                except KeyError:
                    pass
            return
        yield from index._iter_entries_prefix(self, nodes_by_key, keys)

    def key_count(self):
        """Return an estimate of the number of keys in this index.

        For BTreeGraphIndex the estimate is exact as it is contained in the
        header.
        """
        if self._key_count is None:
            self._get_root_node()
        return self._key_count

    def _compute_row_offsets(self):
        """Fill out the _row_offsets attribute based on _row_lengths."""
        offsets = []
        row_offset = 0
        for row in self._row_lengths:
            offsets.append(row_offset)
            row_offset += row
        offsets.append(row_offset)
        self._row_offsets = offsets

    def _parse_header_from_bytes(self, bytes):
        """Parse the header from a region of bytes.

        :param bytes: The data to parse.
        :return: An offset, data tuple such as readv yields, for the unparsed
            data. (which may be of length 0).
        """
        signature = bytes[0:len(self._signature())]
        if not signature == self._signature():
            raise index.BadIndexFormatSignature(self._name, BTreeGraphIndex)
        lines = bytes[len(self._signature()):].splitlines()
        options_line = lines[0]
        if not options_line.startswith(_OPTION_NODE_REFS):
            raise index.BadIndexOptions(self)
        try:
            self.node_ref_lists = int(options_line[len(_OPTION_NODE_REFS):])
        except ValueError:
            raise index.BadIndexOptions(self)
        options_line = lines[1]
        if not options_line.startswith(_OPTION_KEY_ELEMENTS):
            raise index.BadIndexOptions(self)
        try:
            self._key_length = int(options_line[len(_OPTION_KEY_ELEMENTS):])
        except ValueError:
            raise index.BadIndexOptions(self)
        options_line = lines[2]
        if not options_line.startswith(_OPTION_LEN):
            raise index.BadIndexOptions(self)
        try:
            self._key_count = int(options_line[len(_OPTION_LEN):])
        except ValueError:
            raise index.BadIndexOptions(self)
        options_line = lines[3]
        if not options_line.startswith(_OPTION_ROW_LENGTHS):
            raise index.BadIndexOptions(self)
        try:
            self._row_lengths = [int(length) for length in options_line[len(_OPTION_ROW_LENGTHS):].split(b',') if length]
        except ValueError:
            raise index.BadIndexOptions(self)
        self._compute_row_offsets()
        header_end = len(signature) + sum(map(len, lines[0:4])) + 4
        return (header_end, bytes[header_end:])

    def _read_nodes(self, nodes):
        """Read some nodes from disk into the LRU cache.

        This performs a readv to get the node data into memory, and parses each
        node, then yields it to the caller. The nodes are requested in the
        supplied order. If possible doing sort() on the list before requesting
        a read may improve performance.

        :param nodes: The nodes to read. 0 - first node, 1 - second node etc.
        :return: None
        """
        bytes = None
        ranges = []
        base_offset = self._base_offset
        for index in nodes:
            offset = index * _PAGE_SIZE
            size = _PAGE_SIZE
            if index == 0:
                if self._size:
                    size = min(_PAGE_SIZE, self._size)
                else:
                    bytes = self._transport.get_bytes(self._name)
                    num_bytes = len(bytes)
                    self._size = num_bytes - base_offset
                    ranges = [(start, min(_PAGE_SIZE, num_bytes - start)) for start in range(base_offset, num_bytes, _PAGE_SIZE)]
                    break
            else:
                if offset > self._size:
                    raise AssertionError('tried to read past the end of the file %s > %s' % (offset, self._size))
                size = min(size, self._size - offset)
            ranges.append((base_offset + offset, size))
        if not ranges:
            return
        elif bytes is not None:
            data_ranges = [(start, bytes[start:start + size]) for start, size in ranges]
        elif self._file is None:
            data_ranges = self._transport.readv(self._name, ranges)
        else:
            data_ranges = []
            for offset, size in ranges:
                self._file.seek(offset)
                data_ranges.append((offset, self._file.read(size)))
        for offset, data in data_ranges:
            offset -= base_offset
            if offset == 0:
                offset, data = self._parse_header_from_bytes(data)
                if len(data) == 0:
                    continue
            bytes = zlib.decompress(data)
            if bytes.startswith(_LEAF_FLAG):
                node = self._leaf_factory(bytes, self._key_length, self.node_ref_lists)
            elif bytes.startswith(_INTERNAL_FLAG):
                node = _InternalNode(bytes)
            else:
                raise AssertionError('Unknown node type for %r' % bytes)
            yield (offset // _PAGE_SIZE, node)

    def _signature(self):
        """The file signature for this index type."""
        return _BTSIGNATURE

    def validate(self):
        """Validate that everything in the index can be accessed."""
        self._get_root_node()
        if len(self._row_lengths) > 1:
            start_node = self._row_offsets[1]
        else:
            start_node = 1
        node_end = self._row_offsets[-1]
        for node in self._read_nodes(list(range(start_node, node_end))):
            pass