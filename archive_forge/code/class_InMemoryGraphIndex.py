import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class InMemoryGraphIndex(GraphIndexBuilder):
    """A GraphIndex which operates entirely out of memory and is mutable.

    This is designed to allow the accumulation of GraphIndex entries during a
    single write operation, where the accumulated entries need to be immediately
    available - for example via a CombinedGraphIndex.
    """

    def add_nodes(self, nodes):
        """Add nodes to the index.

        :param nodes: An iterable of (key, node_refs, value) entries to add.
        """
        if self.reference_lists:
            for key, value, node_refs in nodes:
                self.add_node(key, value, node_refs)
        else:
            for key, value in nodes:
                self.add_node(key, value)

    def iter_all_entries(self):
        """Iterate over all keys within the index

        :return: An iterable of (index, key, reference_lists, value). There is no
            defined order for the result iteration - it will be in the most
            efficient order for the index (in this case dictionary hash order).
        """
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(3, 'iter_all_entries scales with size of history.')
        if self.reference_lists:
            for key, (absent, references, value) in self._nodes.items():
                if not absent:
                    yield (self, key, value, references)
        else:
            for key, (absent, references, value) in self._nodes.items():
                if not absent:
                    yield (self, key, value)

    def iter_entries(self, keys):
        """Iterate over keys within the index.

        :param keys: An iterable providing the keys to be retrieved.
        :return: An iterable of (index, key, value, reference_lists). There is no
            defined order for the result iteration - it will be in the most
            efficient order for the index (keys iteration order in this case).
        """
        nodes = self._nodes
        keys = [key for key in keys if key in nodes]
        if self.reference_lists:
            for key in keys:
                node = nodes[key]
                if not node[0]:
                    yield (self, key, node[2], node[1])
        else:
            for key in keys:
                node = nodes[key]
                if not node[0]:
                    yield (self, key, node[2])

    def iter_entries_prefix(self, keys):
        """Iterate over keys within the index using prefix matching.

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
        if self._key_length == 1:
            for key in keys:
                _sanity_check_key(self, key)
                node = self._nodes[key]
                if node[0]:
                    continue
                if self.reference_lists:
                    yield (self, key, node[2], node[1])
                else:
                    yield (self, key, node[2])
            return
        nodes_by_key = self._get_nodes_by_key()
        yield from _iter_entries_prefix(self, nodes_by_key, keys)

    def key_count(self):
        """Return an estimate of the number of keys in this index.

        For InMemoryGraphIndex the estimate is exact.
        """
        return len(self._nodes) - len(self._absent_keys)

    def validate(self):
        """In memory index's have no known corruption at the moment."""

    def __lt__(self, other):
        if not isinstance(other, GraphIndex) and (not isinstance(other, InMemoryGraphIndex)):
            raise TypeError(other)
        return hash(self) < hash(other)