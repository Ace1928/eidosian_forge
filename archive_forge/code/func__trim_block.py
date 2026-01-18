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
def _trim_block(self, last_byte):
    """Create a new GroupCompressBlock, with just some of the content."""
    trace.mutter('stripping trailing bytes from groupcompress block %d => %d', self._block._content_length, last_byte)
    new_block = GroupCompressBlock()
    self._block._ensure_content(last_byte)
    new_block.set_content(self._block._content[:last_byte])
    self._block = new_block