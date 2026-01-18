import struct
import zlib
from .static_tuple import StaticTuple
def _deserialise_internal_node(data, key, search_key_func=None):
    global _unknown, _LeafNode, _InternalNode
    if _InternalNode is None:
        from . import chk_map
        _unknown = chk_map._unknown
        _LeafNode = chk_map.LeafNode
        _InternalNode = chk_map.InternalNode
    result = _InternalNode(search_key_func=search_key_func)
    lines = data.split(b'\n')
    if lines[-1] != b'':
        raise ValueError("last line must be ''")
    lines.pop(-1)
    items = {}
    if lines[0] != b'chknode:':
        raise ValueError('not a serialised internal node: %r' % bytes)
    maximum_size = int(lines[1])
    width = int(lines[2])
    length = int(lines[3])
    common_prefix = lines[4]
    for line in lines[5:]:
        line = common_prefix + line
        prefix, flat_key = line.rsplit(b'\x00', 1)
        items[prefix] = StaticTuple(flat_key)
    if len(items) == 0:
        raise AssertionError("We didn't find any item for %s" % key)
    result._items = items
    result._len = length
    result._maximum_size = maximum_size
    result._key = key
    result._key_width = width
    result._raw_size = None
    result._node_width = len(prefix)
    result._search_prefix = common_prefix
    return result