import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
class LeafNode(Node):
    """A node containing actual key:value pairs.

    :ivar _items: A dict of key->value items. The key is in tuple form.
    :ivar _size: The number of bytes that would be used by serializing all of
        the key/value pairs.
    """
    __slots__ = ('_common_serialised_prefix',)

    def __init__(self, search_key_func=None):
        Node.__init__(self)
        self._common_serialised_prefix = None
        if search_key_func is None:
            self._search_key_func = _search_key_plain
        else:
            self._search_key_func = search_key_func

    def __repr__(self):
        items_str = str(sorted(self._items))
        if len(items_str) > 20:
            items_str = items_str[:16] + '...]'
        return '%s(key:%s len:%s size:%s max:%s prefix:%s keywidth:%s items:%s)' % (self.__class__.__name__, self._key, self._len, self._raw_size, self._maximum_size, self._search_prefix, self._key_width, items_str)

    def _current_size(self):
        """Answer the current serialised size of this node.

        This differs from self._raw_size in that it includes the bytes used for
        the header.
        """
        if self._common_serialised_prefix is None:
            bytes_for_items = 0
            prefix_len = 0
        else:
            prefix_len = len(self._common_serialised_prefix)
            bytes_for_items = self._raw_size - prefix_len * self._len
        return 9 + len(str(self._maximum_size)) + 1 + len(str(self._key_width)) + 1 + len(str(self._len)) + 1 + prefix_len + 1 + bytes_for_items

    @classmethod
    def deserialise(klass, bytes, key, search_key_func=None):
        """Deserialise bytes, with key key, into a LeafNode.

        :param bytes: The bytes of the node.
        :param key: The key that the serialised node has.
        """
        key = expect_static_tuple(key)
        return _deserialise_leaf_node(bytes, key, search_key_func=search_key_func)

    def iteritems(self, store, key_filter=None):
        """Iterate over items in the node.

        :param key_filter: A filter to apply to the node. It should be a
            list/set/dict or similar repeatedly iterable container.
        """
        if key_filter is not None:
            filters = {}
            for key in key_filter:
                if len(key) == self._key_width:
                    try:
                        yield (key, self._items[key])
                    except KeyError:
                        pass
                else:
                    filters.setdefault(len(key), set()).add(key)
            if filters:
                filters_itemview = filters.items()
                for item in self._items.items():
                    for length, length_filter in filters_itemview:
                        if item[0][:length] in length_filter:
                            yield item
                            break
        else:
            yield from self._items.items()

    def _key_value_len(self, key, value):
        return len(self._serialise_key(key)) + 1 + len(b'%d' % value.count(b'\n')) + 1 + len(value) + 1

    def _search_key(self, key):
        return self._search_key_func(key)

    def _map_no_split(self, key, value):
        """Map a key to a value.

        This assumes either the key does not already exist, or you have already
        removed its size and length from self.

        :return: True if adding this node should cause us to split.
        """
        self._items[key] = value
        self._raw_size += self._key_value_len(key, value)
        self._len += 1
        serialised_key = self._serialise_key(key)
        if self._common_serialised_prefix is None:
            self._common_serialised_prefix = serialised_key
        else:
            self._common_serialised_prefix = self.common_prefix(self._common_serialised_prefix, serialised_key)
        search_key = self._search_key(key)
        if self._search_prefix is _unknown:
            self._compute_search_prefix()
        if self._search_prefix is None:
            self._search_prefix = search_key
        else:
            self._search_prefix = self.common_prefix(self._search_prefix, search_key)
        if self._len > 1 and self._maximum_size and (self._current_size() > self._maximum_size):
            if search_key != self._search_prefix or not self._are_search_keys_identical():
                return True
        return False

    def _split(self, store):
        """We have overflowed.

        Split this node into multiple LeafNodes, return it up the stack so that
        the next layer creates a new InternalNode and references the new nodes.

        :return: (common_serialised_prefix, [(node_serialised_prefix, node)])
        """
        if self._search_prefix is _unknown:
            raise AssertionError('Search prefix must be known')
        common_prefix = self._search_prefix
        split_at = len(common_prefix) + 1
        result = {}
        for key, value in self._items.items():
            search_key = self._search_key(key)
            prefix = search_key[:split_at]
            if len(prefix) < split_at:
                prefix += b'\x00' * (split_at - len(prefix))
            if prefix not in result:
                node = LeafNode(search_key_func=self._search_key_func)
                node.set_maximum_size(self._maximum_size)
                node._key_width = self._key_width
                result[prefix] = node
            else:
                node = result[prefix]
            sub_prefix, node_details = node.map(store, key, value)
            if len(node_details) > 1:
                if prefix != sub_prefix:
                    result.pop(prefix)
                new_node = InternalNode(sub_prefix, search_key_func=self._search_key_func)
                new_node.set_maximum_size(self._maximum_size)
                new_node._key_width = self._key_width
                for split, node in node_details:
                    new_node.add_node(split, node)
                result[prefix] = new_node
        return (common_prefix, list(result.items()))

    def map(self, store, key, value):
        """Map key to value."""
        if key in self._items:
            self._raw_size -= self._key_value_len(key, self._items[key])
            self._len -= 1
        self._key = None
        if self._map_no_split(key, value):
            return self._split(store)
        else:
            if self._search_prefix is _unknown:
                raise AssertionError('%r must be known' % self._search_prefix)
            return (self._search_prefix, [(b'', self)])
    _serialise_key = b'\x00'.join

    def serialise(self, store):
        """Serialise the LeafNode to store.

        :param store: A VersionedFiles honouring the CHK extensions.
        :return: An iterable of the keys inserted by this operation.
        """
        lines = [b'chkleaf:\n']
        lines.append(b'%d\n' % self._maximum_size)
        lines.append(b'%d\n' % self._key_width)
        lines.append(b'%d\n' % self._len)
        if self._common_serialised_prefix is None:
            lines.append(b'\n')
            if len(self._items) != 0:
                raise AssertionError('If _common_serialised_prefix is None we should have no items')
        else:
            lines.append(b'%s\n' % (self._common_serialised_prefix,))
            prefix_len = len(self._common_serialised_prefix)
        for key, value in sorted(self._items.items()):
            value_lines = osutils.chunks_to_lines([value + b'\n'])
            serialized = b'%s\x00%d\n' % (self._serialise_key(key), len(value_lines))
            if not serialized.startswith(self._common_serialised_prefix):
                raise AssertionError('We thought the common prefix was %r but entry %r does not have it in common' % (self._common_serialised_prefix, serialized))
            lines.append(serialized[prefix_len:])
            lines.extend(value_lines)
        sha1, _, _ = store.add_lines((None,), (), lines)
        self._key = StaticTuple(b'sha1:' + sha1).intern()
        data = b''.join(lines)
        if len(data) != self._current_size():
            raise AssertionError('Invalid _current_size')
        _get_cache()[self._key] = data
        return [self._key]

    def refs(self):
        """Return the references to other CHK's held by this node."""
        return []

    def _compute_search_prefix(self):
        """Determine the common search prefix for all keys in this node.

        :return: A bytestring of the longest search key prefix that is
            unique within this node.
        """
        search_keys = [self._search_key_func(key) for key in self._items]
        self._search_prefix = self.common_prefix_for_keys(search_keys)
        return self._search_prefix

    def _are_search_keys_identical(self):
        """Check to see if the search keys for all entries are the same.

        When using a hash as the search_key it is possible for non-identical
        keys to collide. If that happens enough, we may try overflow a
        LeafNode, but as all are collisions, we must not split.
        """
        common_search_key = None
        for key in self._items:
            search_key = self._search_key(key)
            if common_search_key is None:
                common_search_key = search_key
            elif search_key != common_search_key:
                return False
        return True

    def _compute_serialised_prefix(self):
        """Determine the common prefix for serialised keys in this node.

        :return: A bytestring of the longest serialised key prefix that is
            unique within this node.
        """
        serialised_keys = [self._serialise_key(key) for key in self._items]
        self._common_serialised_prefix = self.common_prefix_for_keys(serialised_keys)
        return self._common_serialised_prefix

    def unmap(self, store, key):
        """Unmap key from the node."""
        try:
            self._raw_size -= self._key_value_len(key, self._items[key])
        except KeyError:
            trace.mutter('key %s not found in %r', key, self._items)
            raise
        self._len -= 1
        del self._items[key]
        self._key = None
        self._compute_search_prefix()
        self._compute_serialised_prefix()
        return self