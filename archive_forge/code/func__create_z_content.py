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
def _create_z_content(self):
    if self._z_content_chunks is not None:
        return
    if self._content_chunks is not None:
        chunks = self._content_chunks
    else:
        chunks = (self._content,)
    self._create_z_content_from_chunks(chunks)