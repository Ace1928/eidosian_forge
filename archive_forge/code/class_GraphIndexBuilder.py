import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class GraphIndexBuilder:
    """A builder that can build a GraphIndex.

    The resulting graph has the structure::

      _SIGNATURE OPTIONS NODES NEWLINE
      _SIGNATURE     := 'Bazaar Graph Index 1' NEWLINE
      OPTIONS        := 'node_ref_lists=' DIGITS NEWLINE
      NODES          := NODE*
      NODE           := KEY NULL ABSENT? NULL REFERENCES NULL VALUE NEWLINE
      KEY            := Not-whitespace-utf8
      ABSENT         := 'a'
      REFERENCES     := REFERENCE_LIST (TAB REFERENCE_LIST){node_ref_lists - 1}
      REFERENCE_LIST := (REFERENCE (CR REFERENCE)*)?
      REFERENCE      := DIGITS  ; digits is the byte offset in the index of the
                                ; referenced key.
      VALUE          := no-newline-no-null-bytes
    """

    def __init__(self, reference_lists=0, key_elements=1):
        """Create a GraphIndex builder.

        :param reference_lists: The number of node references lists for each
            entry.
        :param key_elements: The number of bytestrings in each key.
        """
        self.reference_lists = reference_lists
        self._nodes = {}
        self._absent_keys = set()
        self._nodes_by_key = None
        self._key_length = key_elements
        self._optimize_for_size = False
        self._combine_backing_indices = True

    def _check_key(self, key):
        """Raise BadIndexKey if key is not a valid key for this index."""
        if type(key) not in (tuple, StaticTuple):
            raise BadIndexKey(key)
        if self._key_length != len(key):
            raise BadIndexKey(key)
        for element in key:
            if not element or type(element) != bytes or _whitespace_re.search(element) is not None:
                raise BadIndexKey(key)

    def _external_references(self):
        """Return references that are not present in this index.
        """
        keys = set()
        refs = set()
        if self.reference_lists > 1:
            for node in self.iter_all_entries():
                keys.add(node[1])
                refs.update(node[3][1])
            return refs - keys
        else:
            return set()

    def _get_nodes_by_key(self):
        if self._nodes_by_key is None:
            nodes_by_key = {}
            if self.reference_lists:
                for key, (absent, references, value) in self._nodes.items():
                    if absent:
                        continue
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = (key, value, references)
            else:
                for key, (absent, references, value) in self._nodes.items():
                    if absent:
                        continue
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = (key, value)
            self._nodes_by_key = nodes_by_key
        return self._nodes_by_key

    def _update_nodes_by_key(self, key, value, node_refs):
        """Update the _nodes_by_key dict with a new key.

        For a key of (foo, bar, baz) create
        _nodes_by_key[foo][bar][baz] = key_value
        """
        if self._nodes_by_key is None:
            return
        key_dict = self._nodes_by_key
        if self.reference_lists:
            key_value = StaticTuple(key, value, node_refs)
        else:
            key_value = StaticTuple(key, value)
        for subkey in key[:-1]:
            key_dict = key_dict.setdefault(subkey, {})
        key_dict[key[-1]] = key_value

    def _check_key_ref_value(self, key, references, value):
        """Check that 'key' and 'references' are all valid.

        :param key: A key tuple. Must conform to the key interface (be a tuple,
            be of the right length, not have any whitespace or nulls in any key
            element.)
        :param references: An iterable of reference lists. Something like
            [[(ref, key)], [(ref, key), (other, key)]]
        :param value: The value associate with this key. Must not contain
            newlines or null characters.
        :return: (node_refs, absent_references)

            * node_refs: basically a packed form of 'references' where all
              iterables are tuples
            * absent_references: reference keys that are not in self._nodes.
              This may contain duplicates if the same key is referenced in
              multiple lists.
        """
        as_st = StaticTuple.from_sequence
        self._check_key(key)
        if _newline_null_re.search(value) is not None:
            raise BadIndexValue(value)
        if len(references) != self.reference_lists:
            raise BadIndexValue(references)
        node_refs = []
        absent_references = []
        for reference_list in references:
            for reference in reference_list:
                if reference not in self._nodes:
                    self._check_key(reference)
                    absent_references.append(reference)
            reference_list = as_st([as_st(ref).intern() for ref in reference_list])
            node_refs.append(reference_list)
        return (as_st(node_refs), absent_references)

    def add_node(self, key, value, references=()):
        """Add a node to the index.

        :param key: The key. keys are non-empty tuples containing
            as many whitespace-free utf8 bytestrings as the key length
            defined for this index.
        :param references: An iterable of iterables of keys. Each is a
            reference to another key.
        :param value: The value to associate with the key. It may be any
            bytes as long as it does not contain \\0 or \\n.
        """
        node_refs, absent_references = self._check_key_ref_value(key, references, value)
        if key in self._nodes and self._nodes[key][0] != b'a':
            raise BadIndexDuplicateKey(key, self)
        for reference in absent_references:
            self._nodes[reference] = (b'a', (), b'')
        self._absent_keys.update(absent_references)
        self._absent_keys.discard(key)
        self._nodes[key] = (b'', node_refs, value)
        if self._nodes_by_key is not None and self._key_length > 1:
            self._update_nodes_by_key(key, value, node_refs)

    def clear_cache(self):
        """See GraphIndex.clear_cache()

        This is a no-op, but we need the api to conform to a generic 'Index'
        abstraction.
        """

    def finish(self):
        """Finish the index.

        :returns: cBytesIO holding the full context of the index as it
        should be written to disk.
        """
        lines = [_SIGNATURE]
        lines.append(b'%s%d\n' % (_OPTION_NODE_REFS, self.reference_lists))
        lines.append(b'%s%d\n' % (_OPTION_KEY_ELEMENTS, self._key_length))
        key_count = len(self._nodes) - len(self._absent_keys)
        lines.append(b'%s%d\n' % (_OPTION_LEN, key_count))
        prefix_length = sum((len(x) for x in lines))
        nodes = sorted(self._nodes.items())
        expected_bytes = None
        if self.reference_lists:
            key_offset_info = []
            non_ref_bytes = prefix_length
            total_references = 0
            for key, (absent, references, value) in nodes:
                key_offset_info.append((key, non_ref_bytes, total_references))
                non_ref_bytes += sum((len(element) for element in key))
                if self._key_length > 1:
                    non_ref_bytes += self._key_length - 1
                non_ref_bytes += len(value) + 3 + 1
                if absent:
                    non_ref_bytes += 1
                elif self.reference_lists:
                    non_ref_bytes += self.reference_lists - 1
                    for ref_list in references:
                        total_references += len(ref_list)
                        if ref_list:
                            non_ref_bytes += len(ref_list) - 1
            digits = 1
            possible_total_bytes = non_ref_bytes + total_references * digits
            while 10 ** digits < possible_total_bytes:
                digits += 1
                possible_total_bytes = non_ref_bytes + total_references * digits
            expected_bytes = possible_total_bytes + 1
            key_addresses = {}
            for key, non_ref_bytes, total_references in key_offset_info:
                key_addresses[key] = non_ref_bytes + total_references * digits
            format_string = b'%%0%dd' % digits
        for key, (absent, references, value) in nodes:
            flattened_references = []
            for ref_list in references:
                ref_addresses = []
                for reference in ref_list:
                    ref_addresses.append(format_string % key_addresses[reference])
                flattened_references.append(b'\r'.join(ref_addresses))
            string_key = b'\x00'.join(key)
            lines.append(b'%s\x00%s\x00%s\x00%s\n' % (string_key, absent, b'\t'.join(flattened_references), value))
        lines.append(b'\n')
        result = BytesIO(b''.join(lines))
        if expected_bytes and len(result.getvalue()) != expected_bytes:
            raise errors.BzrError('Failed index creation. Internal error: mismatched output length and expected length: %d %d' % (len(result.getvalue()), expected_bytes))
        return result

    def set_optimize(self, for_size=None, combine_backing_indices=None):
        """Change how the builder tries to optimize the result.

        :param for_size: Tell the builder to try and make the index as small as
            possible.
        :param combine_backing_indices: If the builder spills to disk to save
            memory, should the on-disk indices be combined. Set to True if you
            are going to be probing the index, but to False if you are not. (If
            you are not querying, then the time spent combining is wasted.)
        :return: None
        """
        if for_size is not None:
            self._optimize_for_size = for_size
        if combine_backing_indices is not None:
            self._combine_backing_indices = combine_backing_indices

    def find_ancestry(self, keys, ref_list_num):
        """See CombinedGraphIndex.find_ancestry()"""
        pending = set(keys)
        parent_map = {}
        missing_keys = set()
        while pending:
            next_pending = set()
            for _, key, value, ref_lists in self.iter_entries(pending):
                parent_keys = ref_lists[ref_list_num]
                parent_map[key] = parent_keys
                next_pending.update([p for p in parent_keys if p not in parent_map])
                missing_keys.update(pending.difference(parent_map))
            pending = next_pending
        return (parent_map, missing_keys)