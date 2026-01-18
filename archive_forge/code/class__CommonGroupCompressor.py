import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
class _CommonGroupCompressor:

    def __init__(self, settings=None):
        """Create a GroupCompressor."""
        self.chunks = []
        self._last = None
        self.endpoint = 0
        self.input_bytes = 0
        self.labels_deltas = {}
        self._delta_index = None
        self._block = GroupCompressBlock()
        if settings is None:
            self._settings = {}
        else:
            self._settings = settings

    def compress(self, key, chunks, length, expected_sha, nostore_sha=None, soft=False):
        """Compress lines with label key.

        :param key: A key tuple. It is stored in the output
            for identification of the text during decompression. If the last
            element is b'None' it is replaced with the sha1 of the text -
            e.g. sha1:xxxxxxx.
        :param chunks: Chunks of bytes to be compressed
        :param length: Length of chunks
        :param expected_sha: If non-None, the sha the lines are believed to
            have. During compression the sha is calculated; a mismatch will
            cause an error.
        :param nostore_sha: If the computed sha1 sum matches, we will raise
            ExistingContent rather than adding the text.
        :param soft: Do a 'soft' compression. This means that we require larger
            ranges to match to be considered for a copy command.

        :return: The sha1 of lines, the start and end offsets in the delta, and
            the type ('fulltext' or 'delta').

        :seealso VersionedFiles.add_lines:
        """
        if length == 0:
            if nostore_sha == _null_sha1:
                raise ExistingContent()
            return (_null_sha1, 0, 0, 'fulltext')
        if expected_sha is not None:
            sha1 = expected_sha
        else:
            sha1 = osutils.sha_strings(chunks)
        if nostore_sha is not None:
            if sha1 == nostore_sha:
                raise ExistingContent()
        if key[-1] is None:
            key = key[:-1] + (b'sha1:' + sha1,)
        start, end, type = self._compress(key, chunks, length, length / 2, soft)
        return (sha1, start, end, type)

    def _compress(self, key, chunks, input_len, max_delta_size, soft=False):
        """Compress lines with label key.

        :param key: A key tuple. It is stored in the output for identification
            of the text during decompression.

        :param chunks: The chunks of bytes to be compressed

        :param input_len: The length of the chunks

        :param max_delta_size: The size above which we issue a fulltext instead
            of a delta.

        :param soft: Do a 'soft' compression. This means that we require larger
            ranges to match to be considered for a copy command.

        :return: The sha1 of lines, the start and end offsets in the delta, and
            the type ('fulltext' or 'delta').
        """
        raise NotImplementedError(self._compress)

    def extract(self, key):
        """Extract a key previously added to the compressor.

        :param key: The key to extract.
        :return: An iterable over chunks and the sha1.
        """
        start_byte, start_chunk, end_byte, end_chunk = self.labels_deltas[key]
        delta_chunks = self.chunks[start_chunk:end_chunk]
        stored_bytes = b''.join(delta_chunks)
        kind = stored_bytes[:1]
        if kind == b'f':
            fulltext_len, offset = decode_base128_int(stored_bytes[1:10])
            data_len = fulltext_len + 1 + offset
            if data_len != len(stored_bytes):
                raise ValueError('Index claimed fulltext len, but stored bytes claim %s != %s' % (len(stored_bytes), data_len))
            data = [stored_bytes[offset + 1:]]
        else:
            if kind != b'd':
                raise ValueError('Unknown content kind, bytes claim %s' % kind)
            source = b''.join(self.chunks[:start_chunk])
            delta_len, offset = decode_base128_int(stored_bytes[1:10])
            data_len = delta_len + 1 + offset
            if data_len != len(stored_bytes):
                raise ValueError('Index claimed delta len, but stored bytes claim %s != %s' % (len(stored_bytes), data_len))
            data = [apply_delta(source, stored_bytes[offset + 1:])]
        data_sha1 = osutils.sha_strings(data)
        return (data, data_sha1)

    def flush(self):
        """Finish this group, creating a formatted stream.

        After calling this, the compressor should no longer be used
        """
        self._block.set_chunked_content(self.chunks, self.endpoint)
        self.chunks = None
        self._delta_index = None
        return self._block

    def pop_last(self):
        """Call this if you want to 'revoke' the last compression.

        After this, the data structures will be rolled back, but you cannot do
        more compression.
        """
        self._delta_index = None
        del self.chunks[self._last[0]:]
        self.endpoint = self._last[1]
        self._last = None

    def ratio(self):
        """Return the overall compression ratio."""
        return float(self.input_bytes) / float(self.endpoint)