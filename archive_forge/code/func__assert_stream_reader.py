from gitdb.test.lib import (
from gitdb import (
from gitdb.util import hex_to_bin
import zlib
from gitdb.typ import (
import tempfile
import os
from io import BytesIO
def _assert_stream_reader(self, stream, cdata, rewind_stream=lambda s: None):
    """Make stream tests - the orig_stream is seekable, allowing it to be
        rewound and reused
        :param cdata: the data we expect to read from stream, the contents
        :param rewind_stream: function called to rewind the stream to make it ready
            for reuse"""
    ns = 10
    assert len(cdata) > ns - 1, 'Data must be larger than %i, was %i' % (ns, len(cdata))
    ss = len(cdata) // ns
    for i in range(ns):
        data = stream.read(ss)
        chunk = cdata[i * ss:(i + 1) * ss]
        assert data == chunk
    rest = stream.read()
    if rest:
        assert rest == cdata[-len(rest):]
    if isinstance(stream, DecompressMemMapReader):
        assert len(stream.data()) == stream.compressed_bytes_read()
    rewind_stream(stream)
    rdata = stream.read()
    assert rdata == cdata
    if isinstance(stream, DecompressMemMapReader):
        assert len(stream.data()) == stream.compressed_bytes_read()