from gitdb.test.lib import (
from gitdb.stream import DeltaApplyReader
from gitdb.pack import (
from gitdb.base import (
from gitdb.fun import delta_types
from gitdb.exc import UnsupportedOperation
from gitdb.util import to_bin_sha
import pytest
import os
import tempfile
def _assert_pack_file(self, pack, version, size):
    assert pack.version() == 2
    assert pack.size() == size
    assert len(pack.checksum()) == 20
    num_obj = 0
    for obj in pack.stream_iter():
        num_obj += 1
        info = pack.info(obj.pack_offset)
        stream = pack.stream(obj.pack_offset)
        assert info.pack_offset == stream.pack_offset
        assert info.type_id == stream.type_id
        assert hasattr(stream, 'read')
        assert obj.read() == stream.read()
        streams = pack.collect_streams(obj.pack_offset)
        assert streams
        try:
            dstream = DeltaApplyReader.new(streams)
        except ValueError:
            continue
        data = dstream.read()
        assert len(data) == dstream.size
        dstream.seek(0)
        assert dstream.read() == data
    assert num_obj == size