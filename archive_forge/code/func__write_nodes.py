from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _write_nodes(self, node_iterator, allow_optimize=True):
    """Write node_iterator out as a B+Tree.

        :param node_iterator: An iterator of sorted nodes. Each node should
            match the output given by iter_all_entries.
        :param allow_optimize: If set to False, prevent setting the optimize
            flag when writing out. This is used by the _spill_mem_keys_to_disk
            functionality.
        :return: A file handle for a temporary file containing a B+Tree for
            the nodes.
        """
    rows = []
    key_count = 0
    self.row_lengths = []
    for node in node_iterator:
        if key_count == 0:
            rows.append(_LeafBuilderRow())
        key_count += 1
        string_key, line = _btree_serializer._flatten_node(node, self.reference_lists)
        self._add_key(string_key, line, rows, allow_optimize=allow_optimize)
    for row in reversed(rows):
        pad = not isinstance(row, _LeafBuilderRow)
        row.finish_node(pad=pad)
    lines = [_BTSIGNATURE]
    lines.append(b'%s%d\n' % (_OPTION_NODE_REFS, self.reference_lists))
    lines.append(b'%s%d\n' % (_OPTION_KEY_ELEMENTS, self._key_length))
    lines.append(b'%s%d\n' % (_OPTION_LEN, key_count))
    row_lengths = [row.nodes for row in rows]
    lines.append(_OPTION_ROW_LENGTHS + ','.join(map(str, row_lengths)).encode('ascii') + b'\n')
    if row_lengths and row_lengths[-1] > 1:
        result = tempfile.NamedTemporaryFile(prefix='bzr-index-')
    else:
        result = BytesIO()
    result.writelines(lines)
    position = sum(map(len, lines))
    if position > _RESERVED_HEADER_BYTES:
        raise AssertionError('Could not fit the header in the reserved space: %d > %d' % (position, _RESERVED_HEADER_BYTES))
    for row in rows:
        reserved = _RESERVED_HEADER_BYTES
        row.spool.flush()
        row.spool.seek(0)
        node = row.spool.read(_PAGE_SIZE)
        result.write(node[reserved:])
        if len(node) == _PAGE_SIZE:
            result.write(b'\x00' * (reserved - position))
        position = 0
        copied_len = osutils.pumpfile(row.spool, result)
        if copied_len != (row.nodes - 1) * _PAGE_SIZE:
            if not isinstance(row, _LeafBuilderRow):
                raise AssertionError('Incorrect amount of data copied expected: %d, got: %d' % ((row.nodes - 1) * _PAGE_SIZE, copied_len))
    result.flush()
    size = result.tell()
    result.seek(0)
    return (result, size)