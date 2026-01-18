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
def _create_z_content_from_chunks(self, chunks):
    compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION)
    compressed_chunks = list(map(compressor.compress, chunks))
    compressed_chunks.append(compressor.flush())
    self._z_content_chunks = [c for c in compressed_chunks if c]
    self._z_content_length = sum(map(len, self._z_content_chunks))