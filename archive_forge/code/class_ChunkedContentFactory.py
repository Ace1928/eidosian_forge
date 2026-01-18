import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
class ChunkedContentFactory(ContentFactory):
    """Static data content factory.

    This takes a 'chunked' list of strings. The only requirement on 'chunked' is
    that ''.join(lines) becomes a valid fulltext. A tuple of a single string
    satisfies this, as does a list of lines.

    :ivar sha1: None, or the sha1 of the content fulltext.
    :ivar size: None, or the size of the content fulltext.
    :ivar storage_kind: The native storage kind of this factory. Always
        'chunked'
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: A tuple of parent keys for self.key. If the object has
        no parent information, None (as opposed to () for an empty list of
        parents).
    :ivar chunks_are_lines: Whether chunks are lines.
     """

    def __init__(self, key, parents, sha1, chunks, chunks_are_lines=None):
        """Create a ContentFactory."""
        self.sha1 = sha1
        self.size = sum(map(len, chunks))
        self.storage_kind = 'chunked'
        self.key = key
        self.parents = parents
        self._chunks = chunks
        self._chunks_are_lines = chunks_are_lines

    def get_bytes_as(self, storage_kind):
        if storage_kind == 'chunked':
            return self._chunks
        elif storage_kind == 'fulltext':
            return b''.join(self._chunks)
        elif storage_kind == 'lines':
            if self._chunks_are_lines:
                return self._chunks
            return list(osutils.chunks_to_lines(self._chunks))
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        if storage_kind == 'chunked':
            return iter(self._chunks)
        elif storage_kind == 'lines':
            if self._chunks_are_lines:
                return iter(self._chunks)
            return iter(osutils.chunks_to_lines(self._chunks))
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)