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
def add_factory(self, key, parents, start, end):
    if not self._factories:
        first = True
    else:
        first = False
    factory = _LazyGroupCompressFactory(key, parents, self, start, end, first=first)
    if end > self._last_byte:
        self._last_byte = end
    self._factories.append(factory)