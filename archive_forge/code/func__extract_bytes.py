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
def _extract_bytes(self):
    try:
        self._manager._prepare_for_extract()
    except zlib.error as value:
        raise DecompressCorruption('zlib: ' + str(value))
    block = self._manager._block
    self._chunks = block.extract(self.key, self._start, self._end)