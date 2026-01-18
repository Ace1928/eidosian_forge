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
class NoDupeAddLinesDecorator:
    """Decorator for a VersionedFiles that skips doing an add_lines if the key
    is already present.
    """

    def __init__(self, store):
        self._store = store

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """See VersionedFiles.add_lines.

        This implementation may return None as the third element of the return
        value when the original store wouldn't.
        """
        if nostore_sha:
            raise NotImplementedError('NoDupeAddLinesDecorator.add_lines does not implement the nostore_sha behaviour.')
        if key[-1] is None:
            sha1 = osutils.sha_strings(lines)
            key = (b'sha1:' + sha1,)
        else:
            sha1 = None
        if key in self._store.get_parent_map([key]):
            if sha1 is None:
                sha1 = osutils.sha_strings(lines)
            return (sha1, sum(map(len, lines)), None)
        return self._store.add_lines(key, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)

    def __getattr__(self, name):
        return getattr(self._store, name)