import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def _object(self, sha, as_stream, index=-1):
    """:return: OInfo or OStream object providing information about the given sha
        :param index: if not -1, its assumed to be the sha's index in the IndexFile"""
    if index < 0:
        index = self._sha_to_index(sha)
    if sha is None:
        sha = self._index.sha(index)
    offset = self._index.offset(index)
    type_id, uncomp_size, data_rela_offset = pack_object_header_info(self._pack._cursor.use_region(offset).buffer())
    if as_stream:
        if type_id not in delta_types:
            packstream = self._pack.stream(offset)
            return OStream(sha, packstream.type, packstream.size, packstream.stream)
        streams = self.collect_streams_at_offset(offset)
        dstream = DeltaApplyReader.new(streams)
        return ODeltaStream(sha, dstream.type, None, dstream)
    else:
        if type_id not in delta_types:
            return OInfo(sha, type_id_to_type_map[type_id], uncomp_size)
        streams = self.collect_streams_at_offset(offset)
        buf = streams[0].read(512)
        offset, src_size = msb_size(buf)
        offset, target_size = msb_size(buf, offset)
        if streams[-1].type_id in delta_types:
            raise BadObject(sha, 'Could not resolve delta object')
        return OInfo(sha, streams[-1].type, target_size)