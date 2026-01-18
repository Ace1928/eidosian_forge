from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _add_key(self, string_key, line, rows, allow_optimize=True):
    """Add a key to the current chunk.

        :param string_key: The key to add.
        :param line: The fully serialised key and value.
        :param allow_optimize: If set to False, prevent setting the optimize
            flag when writing out. This is used by the _spill_mem_keys_to_disk
            functionality.
        """
    new_leaf = False
    if rows[-1].writer is None:
        new_leaf = True
        for pos, internal_row in enumerate(rows[:-1]):
            if internal_row.writer is None:
                length = _PAGE_SIZE
                if internal_row.nodes == 0:
                    length -= _RESERVED_HEADER_BYTES
                if allow_optimize:
                    optimize_for_size = self._optimize_for_size
                else:
                    optimize_for_size = False
                internal_row.writer = chunk_writer.ChunkWriter(length, 0, optimize_for_size=optimize_for_size)
                internal_row.writer.write(_INTERNAL_FLAG)
                internal_row.writer.write(_INTERNAL_OFFSET + b'%d\n' % rows[pos + 1].nodes)
        length = _PAGE_SIZE
        if rows[-1].nodes == 0:
            length -= _RESERVED_HEADER_BYTES
        rows[-1].writer = chunk_writer.ChunkWriter(length, optimize_for_size=self._optimize_for_size)
        rows[-1].writer.write(_LEAF_FLAG)
    if rows[-1].writer.write(line):
        if new_leaf:
            raise index.BadIndexKey(string_key)
        rows[-1].finish_node()
        key_line = string_key + b'\n'
        new_row = True
        for row in reversed(rows[:-1]):
            if row.writer.write(key_line):
                row.finish_node()
            else:
                new_row = False
                break
        if new_row:
            if 'index' in debug.debug_flags:
                trace.mutter('Inserting new global row.')
            new_row = _InternalBuilderRow()
            reserved_bytes = 0
            rows.insert(0, new_row)
            new_row.writer = chunk_writer.ChunkWriter(_PAGE_SIZE - _RESERVED_HEADER_BYTES, reserved_bytes, optimize_for_size=self._optimize_for_size)
            new_row.writer.write(_INTERNAL_FLAG)
            new_row.writer.write(_INTERNAL_OFFSET + b'%d\n' % (rows[1].nodes - 1))
            new_row.writer.write(key_line)
        self._add_key(string_key, line, rows, allow_optimize=allow_optimize)