import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _seek_and_read(self, fp, offsets, relpath='<unknown>'):
    """An implementation of readv that uses fp.seek and fp.read.

        This uses _coalesce_offsets to issue larger reads and fewer seeks.

        :param fp: A file-like object that supports seek() and read(size).
            Note that implementations are allowed to call .close() on this file
            handle, so don't trust that you can use it for other work.
        :param offsets: A list of offsets to be read from the given file.
        :return: yield (pos, data) tuples for each request
        """
    offsets = list(offsets)
    sorted_offsets = sorted(offsets)
    offset_stack = iter(offsets)
    cur_offset_and_size = next(offset_stack)
    coalesced = self._coalesce_offsets(sorted_offsets, limit=self._max_readv_combine, fudge_factor=self._bytes_to_read_before_seek)
    data_map = {}
    try:
        for c_offset in coalesced:
            fp.seek(c_offset.start)
            data = fp.read(c_offset.length)
            if len(data) < c_offset.length:
                raise errors.ShortReadvError(relpath, c_offset.start, c_offset.length, actual=len(data))
            for suboffset, subsize in c_offset.ranges:
                key = (c_offset.start + suboffset, subsize)
                data_map[key] = data[suboffset:suboffset + subsize]
            while cur_offset_and_size in data_map:
                this_data = data_map.pop(cur_offset_and_size)
                this_offset = cur_offset_and_size[0]
                try:
                    cur_offset_and_size = next(offset_stack)
                except StopIteration:
                    fp.close()
                    cur_offset_and_size = None
                yield (this_offset, this_data)
    finally:
        fp.close()