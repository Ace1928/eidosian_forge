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
class GroupCompressBlock:
    """An object which maintains the internal structure of the compressed data.

    This tracks the meta info (start of text, length, type, etc.)
    """
    GCB_HEADER = b'gcb1z\n'
    GCB_LZ_HEADER = b'gcb1l\n'
    GCB_KNOWN_HEADERS = (GCB_HEADER, GCB_LZ_HEADER)

    def __init__(self):
        self._compressor_name = None
        self._z_content_chunks = None
        self._z_content_decompressor = None
        self._z_content_length = None
        self._content_length = None
        self._content = None
        self._content_chunks = None

    def __len__(self):
        return self._content_length + self._z_content_length

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

    def _parse_bytes(self, data, pos):
        """Read the various lengths from the header.

        This also populates the various 'compressed' buffers.

        :return: The position in bytes just after the last newline
        """
        pos2 = data.index(b'\n', pos, pos + 14)
        self._z_content_length = int(data[pos:pos2])
        pos = pos2 + 1
        pos2 = data.index(b'\n', pos, pos + 14)
        self._content_length = int(data[pos:pos2])
        pos = pos2 + 1
        if len(data) != pos + self._z_content_length:
            raise AssertionError('Invalid bytes: (%d) != %d + %d' % (len(data), pos, self._z_content_length))
        self._z_content_chunks = (data[pos:],)

    @property
    def _z_content(self):
        """Return z_content_chunks as a simple string.

        Meant only to be used by the test suite.
        """
        if self._z_content_chunks is not None:
            return b''.join(self._z_content_chunks)
        return None

    @classmethod
    def from_bytes(cls, bytes):
        out = cls()
        header = bytes[:6]
        if header not in cls.GCB_KNOWN_HEADERS:
            raise ValueError('bytes did not start with any of %r' % (cls.GCB_KNOWN_HEADERS,))
        if header == cls.GCB_HEADER:
            out._compressor_name = 'zlib'
        elif header == cls.GCB_LZ_HEADER:
            out._compressor_name = 'lzma'
        else:
            raise ValueError('unknown compressor: {!r}'.format(header))
        out._parse_bytes(bytes, 6)
        return out

    def extract(self, key, start, end, sha1=None):
        """Extract the text for a specific key.

        :param key: The label used for this content
        :param sha1: TODO (should we validate only when sha1 is supplied?)
        :return: The bytes for the content
        """
        if start == end == 0:
            return []
        self._ensure_content(end)
        c = self._content[start:start + 1]
        if c == b'f':
            type = 'fulltext'
        else:
            if c != b'd':
                raise ValueError('Unknown content control code: %s' % (c,))
            type = 'delta'
        content_len, len_len = decode_base128_int(self._content[start + 1:start + 6])
        content_start = start + 1 + len_len
        if end != content_start + content_len:
            raise ValueError('end != len according to field header %s != %s' % (end, content_start + content_len))
        if c == b'f':
            return [self._content[content_start:end]]
        return [apply_delta_to_source(self._content, content_start, end)]

    def set_chunked_content(self, content_chunks, length):
        """Set the content of this block to the given chunks."""
        self._content_length = length
        self._content_chunks = content_chunks
        self._content = None
        self._z_content_chunks = None

    def set_content(self, content):
        """Set the content of this block."""
        self._content_length = len(content)
        self._content = content
        self._z_content_chunks = None

    def _create_z_content_from_chunks(self, chunks):
        compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION)
        compressed_chunks = list(map(compressor.compress, chunks))
        compressed_chunks.append(compressor.flush())
        self._z_content_chunks = [c for c in compressed_chunks if c]
        self._z_content_length = sum(map(len, self._z_content_chunks))

    def _create_z_content(self):
        if self._z_content_chunks is not None:
            return
        if self._content_chunks is not None:
            chunks = self._content_chunks
        else:
            chunks = (self._content,)
        self._create_z_content_from_chunks(chunks)

    def to_chunks(self):
        """Create the byte stream as a series of 'chunks'"""
        self._create_z_content()
        header = self.GCB_HEADER
        chunks = [b'%s%d\n%d\n' % (header, self._z_content_length, self._content_length)]
        chunks.extend(self._z_content_chunks)
        total_len = sum(map(len, chunks))
        return (total_len, chunks)

    def to_bytes(self):
        """Encode the information into a byte stream."""
        total_len, chunks = self.to_chunks()
        return b''.join(chunks)

    def _dump(self, include_text=False):
        """Take this block, and spit out a human-readable structure.

        :param include_text: Inserts also include text bits, chose whether you
            want this displayed in the dump or not.
        :return: A dump of the given block. The layout is something like:
            [('f', length), ('d', delta_length, text_length, [delta_info])]
            delta_info := [('i', num_bytes, text), ('c', offset, num_bytes),
            ...]
        """
        self._ensure_content()
        result = []
        pos = 0
        while pos < self._content_length:
            kind = self._content[pos:pos + 1]
            pos += 1
            if kind not in (b'f', b'd'):
                raise ValueError('invalid kind character: {!r}'.format(kind))
            content_len, len_len = decode_base128_int(self._content[pos:pos + 5])
            pos += len_len
            if content_len + pos > self._content_length:
                raise ValueError('invalid content_len %d for record @ pos %d' % (content_len, pos - len_len - 1))
            if kind == b'f':
                if include_text:
                    text = self._content[pos:pos + content_len]
                    result.append((b'f', content_len, text))
                else:
                    result.append((b'f', content_len))
            elif kind == b'd':
                delta_content = self._content[pos:pos + content_len]
                delta_info = []
                decomp_len, delta_pos = decode_base128_int(delta_content)
                result.append((b'd', content_len, decomp_len, delta_info))
                measured_len = 0
                while delta_pos < content_len:
                    c = delta_content[delta_pos]
                    delta_pos += 1
                    if c & 128:
                        offset, length, delta_pos = decode_copy_instruction(delta_content, c, delta_pos)
                        if include_text:
                            text = self._content[offset:offset + length]
                            delta_info.append((b'c', offset, length, text))
                        else:
                            delta_info.append((b'c', offset, length))
                        measured_len += length
                    else:
                        if include_text:
                            txt = delta_content[delta_pos:delta_pos + c]
                        else:
                            txt = b''
                        delta_info.append((b'i', c, txt))
                        measured_len += c
                        delta_pos += c
                if delta_pos != content_len:
                    raise ValueError('Delta consumed a bad number of bytes: %d != %d' % (delta_pos, content_len))
                if measured_len != decomp_len:
                    raise ValueError('Delta claimed fulltext was %d bytes, but extraction resulted in %d bytes' % (decomp_len, measured_len))
            pos += content_len
        return result