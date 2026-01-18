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
def _output_chunks(self, new_chunks):
    """Output some chunks.

        :param new_chunks: The chunks to output.
        """
    self._last = (len(self.chunks), self.endpoint)
    endpoint = self.endpoint
    self.chunks.extend(new_chunks)
    endpoint += sum(map(len, new_chunks))
    self.endpoint = endpoint