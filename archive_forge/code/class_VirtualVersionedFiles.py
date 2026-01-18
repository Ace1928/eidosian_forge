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
class VirtualVersionedFiles(VersionedFiles):
    """Dummy implementation for VersionedFiles that uses other functions for
    obtaining fulltexts and parent maps.

    This is always on the bottom of the stack and uses string keys
    (rather than tuples) internally.
    """

    def __init__(self, get_parent_map, get_lines):
        """Create a VirtualVersionedFiles.

        :param get_parent_map: Same signature as Repository.get_parent_map.
        :param get_lines: Should return lines for specified key or None if
                          not available.
        """
        super().__init__()
        self._get_parent_map = get_parent_map
        self._get_lines = get_lines

    def check(self, progressbar=None):
        """See VersionedFiles.check.

        :note: Always returns True for VirtualVersionedFiles.
        """
        return True

    def add_mpdiffs(self, records):
        """See VersionedFiles.mpdiffs.

        :note: Not implemented for VirtualVersionedFiles.
        """
        raise NotImplementedError(self.add_mpdiffs)

    def get_parent_map(self, keys):
        """See VersionedFiles.get_parent_map."""
        parent_view = self._get_parent_map((k for k, in keys)).items()
        return {(k,): tuple(((p,) for p in v)) for k, v in parent_view}

    def get_sha1s(self, keys):
        """See VersionedFiles.get_sha1s."""
        ret = {}
        for k, in keys:
            lines = self._get_lines(k)
            if lines is not None:
                if not isinstance(lines, list):
                    raise AssertionError
                ret[k,] = osutils.sha_strings(lines)
        return ret

    def get_record_stream(self, keys, ordering, include_delta_closure):
        """See VersionedFiles.get_record_stream."""
        for k, in list(keys):
            lines = self._get_lines(k)
            if lines is not None:
                if not isinstance(lines, list):
                    raise AssertionError
                yield ChunkedContentFactory((k,), None, sha1=osutils.sha_strings(lines), chunks=lines, chunks_are_lines=True)
            else:
                yield AbsentContentFactory((k,))

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        """See VersionedFile.iter_lines_added_or_present_in_versions()."""
        for i, (key,) in enumerate(keys):
            if pb is not None:
                pb.update('Finding changed lines', i, len(keys))
            for l in self._get_lines(key):
                yield (l, key)