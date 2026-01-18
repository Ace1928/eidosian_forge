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
def _ensure_content(self, num_bytes=None):
    """Make sure that content has been expanded enough.

        :param num_bytes: Ensure that we have extracted at least num_bytes of
            content. If None, consume everything
        """
    if self._content_length is None:
        raise AssertionError('self._content_length should never be None')
    if num_bytes is None:
        num_bytes = self._content_length
    elif self._content_length is not None and num_bytes > self._content_length:
        raise AssertionError('requested num_bytes (%d) > content length (%d)' % (num_bytes, self._content_length))
    if self._content is None:
        if self._content_chunks is not None:
            self._content = b''.join(self._content_chunks)
            self._content_chunks = None
    if self._content is None:
        if self._z_content_chunks is None:
            raise AssertionError('No content to decompress')
        z_content = b''.join(self._z_content_chunks)
        if z_content == b'':
            self._content = b''
        elif self._compressor_name == 'lzma':
            import pylzma
            self._content = pylzma.decompress(z_content)
        elif self._compressor_name == 'zlib':
            if num_bytes * 4 > self._content_length * 3:
                num_bytes = self._content_length
                self._content = zlib.decompress(z_content)
            else:
                self._z_content_decompressor = zlib.decompressobj()
                self._content = self._z_content_decompressor.decompress(z_content, num_bytes + _ZLIB_DECOMP_WINDOW)
                if not self._z_content_decompressor.unconsumed_tail:
                    self._z_content_decompressor = None
        else:
            raise AssertionError('Unknown compressor: %r' % self._compressor_name)
    if len(self._content) >= num_bytes:
        return
    if self._z_content_decompressor is None:
        raise AssertionError('No decompressor to decompress %d bytes' % num_bytes)
    remaining_decomp = self._z_content_decompressor.unconsumed_tail
    if not remaining_decomp:
        raise AssertionError('Nothing left to decompress')
    needed_bytes = num_bytes - len(self._content)
    self._content += self._z_content_decompressor.decompress(remaining_decomp, needed_bytes + _ZLIB_DECOMP_WINDOW)
    if len(self._content) < num_bytes:
        raise AssertionError('%d bytes wanted, only %d available' % (num_bytes, len(self._content)))
    if not self._z_content_decompressor.unconsumed_tail:
        self._z_content_decompressor = None