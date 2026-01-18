import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
class InternalNode(Node):
    """A node that contains references to other nodes.

    An InternalNode is responsible for mapping search key prefixes to child
    nodes.

    :ivar _items: serialised_key => node dictionary. node may be a tuple,
        LeafNode or InternalNode.
    """
    __slots__ = ('_node_width',)

    def __init__(self, prefix=b'', search_key_func=None):
        Node.__init__(self)
        self._node_width = 0
        self._search_prefix = prefix
        if search_key_func is None:
            self._search_key_func = _search_key_plain
        else:
            self._search_key_func = search_key_func

    def add_node(self, prefix, node):
        """Add a child node with prefix prefix, and node node.

        :param prefix: The search key prefix for node.
        :param node: The node being added.
        """
        if self._search_prefix is None:
            raise AssertionError('_search_prefix should not be None')
        if not prefix.startswith(self._search_prefix):
            raise AssertionError('prefixes mismatch: %s must start with %s' % (prefix, self._search_prefix))
        if len(prefix) != len(self._search_prefix) + 1:
            raise AssertionError('prefix wrong length: len(%s) is not %d' % (prefix, len(self._search_prefix) + 1))
        self._len += len(node)
        if not len(self._items):
            self._node_width = len(prefix)
        if self._node_width != len(self._search_prefix) + 1:
            raise AssertionError('node width mismatch: %d is not %d' % (self._node_width, len(self._search_prefix) + 1))
        self._items[prefix] = node
        self._key = None

    def _current_size(self):
        """Answer the current serialised size of this node."""
        return self._raw_size + len(str(self._len)) + len(str(self._key_width)) + len(str(self._maximum_size))

    @classmethod
    def deserialise(klass, bytes, key, search_key_func=None):
        """Deserialise bytes to an InternalNode, with key key.

        :param bytes: The bytes of the node.
        :param key: The key that the serialised node has.
        :return: An InternalNode instance.
        """
        key = expect_static_tuple(key)
        return _deserialise_internal_node(bytes, key, search_key_func=search_key_func)

    def iteritems(self, store, key_filter=None):
        for node, node_filter in self._iter_nodes(store, key_filter=key_filter):
            yield from node.iteritems(store, key_filter=node_filter)

    def _iter_nodes(self, store, key_filter=None, batch_size=None):
        """Iterate over node objects which match key_filter.

        :param store: A store to use for accessing content.
        :param key_filter: A key filter to filter nodes. Only nodes that might
            contain a key in key_filter will be returned.
        :param batch_size: If not None, then we will return the nodes that had
            to be read using get_record_stream in batches, rather than reading
            them all at once.
        :return: An iterable of nodes. This function does not have to be fully
            consumed.  (There will be no pending I/O when items are being returned.)
        """
        keys = {}
        shortcut = False
        if key_filter is None:
            shortcut = True
            for prefix, node in self._items.items():
                if node.__class__ is StaticTuple:
                    keys[node] = (prefix, None)
                else:
                    yield (node, None)
        elif len(key_filter) == 1:
            for key in key_filter:
                break
            search_prefix = self._search_prefix_filter(key)
            if len(search_prefix) == self._node_width:
                shortcut = True
                try:
                    node = self._items[search_prefix]
                except KeyError:
                    return
                if node.__class__ is StaticTuple:
                    keys[node] = (search_prefix, [key])
                else:
                    yield (node, [key])
                    return
        if not shortcut:
            prefix_to_keys = {}
            length_filters = {}
            for key in key_filter:
                search_prefix = self._search_prefix_filter(key)
                length_filter = length_filters.setdefault(len(search_prefix), set())
                length_filter.add(search_prefix)
                prefix_to_keys.setdefault(search_prefix, []).append(key)
            if self._node_width in length_filters and len(length_filters) == 1:
                search_prefixes = length_filters[self._node_width]
                for search_prefix in search_prefixes:
                    try:
                        node = self._items[search_prefix]
                    except KeyError:
                        continue
                    node_key_filter = prefix_to_keys[search_prefix]
                    if node.__class__ is StaticTuple:
                        keys[node] = (search_prefix, node_key_filter)
                    else:
                        yield (node, node_key_filter)
            else:
                length_filters_itemview = length_filters.items()
                for prefix, node in self._items.items():
                    node_key_filter = []
                    for length, length_filter in length_filters_itemview:
                        sub_prefix = prefix[:length]
                        if sub_prefix in length_filter:
                            node_key_filter.extend(prefix_to_keys[sub_prefix])
                    if node_key_filter:
                        if node.__class__ is StaticTuple:
                            keys[node] = (prefix, node_key_filter)
                        else:
                            yield (node, node_key_filter)
        if keys:
            found_keys = set()
            for key in keys:
                try:
                    bytes = _get_cache()[key]
                except KeyError:
                    continue
                else:
                    node = _deserialise(bytes, key, search_key_func=self._search_key_func)
                    prefix, node_key_filter = keys[key]
                    self._items[prefix] = node
                    found_keys.add(key)
                    yield (node, node_key_filter)
            for key in found_keys:
                del keys[key]
        if keys:
            if batch_size is None:
                batch_size = len(keys)
            key_order = list(keys)
            for batch_start in range(0, len(key_order), batch_size):
                batch = key_order[batch_start:batch_start + batch_size]
                stream = store.get_record_stream(batch, 'unordered', True)
                node_and_filters = []
                for record in stream:
                    bytes = record.get_bytes_as('fulltext')
                    node = _deserialise(bytes, record.key, search_key_func=self._search_key_func)
                    prefix, node_key_filter = keys[record.key]
                    node_and_filters.append((node, node_key_filter))
                    self._items[prefix] = node
                    _get_cache()[record.key] = bytes
                yield from node_and_filters

    def map(self, store, key, value):
        """Map key to value."""
        if not len(self._items):
            raise AssertionError("can't map in an empty InternalNode.")
        search_key = self._search_key(key)
        if self._node_width != len(self._search_prefix) + 1:
            raise AssertionError('node width mismatch: %d is not %d' % (self._node_width, len(self._search_prefix) + 1))
        if not search_key.startswith(self._search_prefix):
            new_prefix = self.common_prefix(self._search_prefix, search_key)
            new_parent = InternalNode(new_prefix, search_key_func=self._search_key_func)
            new_parent.set_maximum_size(self._maximum_size)
            new_parent._key_width = self._key_width
            new_parent.add_node(self._search_prefix[:len(new_prefix) + 1], self)
            return new_parent.map(store, key, value)
        children = [node for node, _ in self._iter_nodes(store, key_filter=[key])]
        if children:
            child = children[0]
        else:
            child = self._new_child(search_key, LeafNode)
        old_len = len(child)
        if isinstance(child, LeafNode):
            old_size = child._current_size()
        else:
            old_size = None
        prefix, node_details = child.map(store, key, value)
        if len(node_details) == 1:
            child = node_details[0][1]
            self._len = self._len - old_len + len(child)
            self._items[search_key] = child
            self._key = None
            new_node = self
            if isinstance(child, LeafNode):
                if old_size is None:
                    trace.mutter('checking remap as InternalNode -> LeafNode')
                    new_node = self._check_remap(store)
                else:
                    new_size = child._current_size()
                    shrinkage = old_size - new_size
                    if shrinkage > 0 and new_size < _INTERESTING_NEW_SIZE or shrinkage > _INTERESTING_SHRINKAGE_LIMIT:
                        trace.mutter('checking remap as size shrunk by %d to be %d', shrinkage, new_size)
                        new_node = self._check_remap(store)
            if new_node._search_prefix is None:
                raise AssertionError('_search_prefix should not be None')
            return (new_node._search_prefix, [(b'', new_node)])
        child = self._new_child(search_key, InternalNode)
        child._search_prefix = prefix
        for split, node in node_details:
            child.add_node(split, node)
        self._len = self._len - old_len + len(child)
        self._key = None
        return (self._search_prefix, [(b'', self)])

    def _new_child(self, search_key, klass):
        """Create a new child node of type klass."""
        child = klass()
        child.set_maximum_size(self._maximum_size)
        child._key_width = self._key_width
        child._search_key_func = self._search_key_func
        self._items[search_key] = child
        return child

    def serialise(self, store):
        """Serialise the node to store.

        :param store: A VersionedFiles honouring the CHK extensions.
        :return: An iterable of the keys inserted by this operation.
        """
        for node in self._items.values():
            if isinstance(node, StaticTuple):
                continue
            if node._key is not None:
                continue
            for key in node.serialise(store):
                yield key
        lines = [b'chknode:\n']
        lines.append(b'%d\n' % self._maximum_size)
        lines.append(b'%d\n' % self._key_width)
        lines.append(b'%d\n' % self._len)
        if self._search_prefix is None:
            raise AssertionError('_search_prefix should not be None')
        lines.append(b'%s\n' % (self._search_prefix,))
        prefix_len = len(self._search_prefix)
        for prefix, node in sorted(self._items.items()):
            if isinstance(node, StaticTuple):
                key = node[0]
            else:
                key = node._key[0]
            serialised = b'%s\x00%s\n' % (prefix, key)
            if not serialised.startswith(self._search_prefix):
                raise AssertionError('prefixes mismatch: %s must start with %s' % (serialised, self._search_prefix))
            lines.append(serialised[prefix_len:])
        sha1, _, _ = store.add_lines((None,), (), lines)
        self._key = StaticTuple(b'sha1:' + sha1).intern()
        _get_cache()[self._key] = b''.join(lines)
        yield self._key

    def _search_key(self, key):
        """Return the serialised key for key in this node."""
        return (self._search_key_func(key) + b'\x00' * self._node_width)[:self._node_width]

    def _search_prefix_filter(self, key):
        """Serialise key for use as a prefix filter in iteritems."""
        return self._search_key_func(key)[:self._node_width]

    def _split(self, offset):
        """Split this node into smaller nodes starting at offset.

        :param offset: The offset to start the new child nodes at.
        :return: An iterable of (prefix, node) tuples. prefix is a byte
            prefix for reaching node.
        """
        if offset >= self._node_width:
            for node in valueview(self._items):
                yield from node._split(offset)

    def refs(self):
        """Return the references to other CHK's held by this node."""
        if self._key is None:
            raise AssertionError('unserialised nodes have no refs.')
        refs = []
        for value in self._items.values():
            if isinstance(value, StaticTuple):
                refs.append(value)
            else:
                refs.append(value.key())
        return refs

    def _compute_search_prefix(self, extra_key=None):
        """Return the unique key prefix for this node.

        :return: A bytestring of the longest search key prefix that is
            unique within this node.
        """
        self._search_prefix = self.common_prefix_for_keys(self._items)
        return self._search_prefix

    def unmap(self, store, key, check_remap=True):
        """Remove key from this node and its children."""
        if not len(self._items):
            raise AssertionError("can't unmap in an empty InternalNode.")
        children = [node for node, _ in self._iter_nodes(store, key_filter=[key])]
        if children:
            child = children[0]
        else:
            raise KeyError(key)
        self._len -= 1
        unmapped = child.unmap(store, key)
        self._key = None
        search_key = self._search_key(key)
        if len(unmapped) == 0:
            del self._items[search_key]
            unmapped = None
        else:
            self._items[search_key] = unmapped
        if len(self._items) == 1:
            return list(self._items.values())[0]
        if isinstance(unmapped, InternalNode):
            return self
        if check_remap:
            return self._check_remap(store)
        else:
            return self

    def _check_remap(self, store):
        """Check if all keys contained by children fit in a single LeafNode.

        :param store: A store to use for reading more nodes
        :return: Either self, or a new LeafNode which should replace self.
        """
        new_leaf = LeafNode(search_key_func=self._search_key_func)
        new_leaf.set_maximum_size(self._maximum_size)
        new_leaf._key_width = self._key_width
        for node, _ in self._iter_nodes(store, batch_size=16):
            if isinstance(node, InternalNode):
                return self
            for key, value in node._items.items():
                if new_leaf._map_no_split(key, value):
                    return self
        trace.mutter('remap generated a new LeafNode')
        return new_leaf