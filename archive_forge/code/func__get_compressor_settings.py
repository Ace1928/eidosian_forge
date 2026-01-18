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
def _get_compressor_settings(self):
    from ..config import GlobalConfig
    if self._max_bytes_to_index is None:
        c = GlobalConfig()
        val = c.get_user_option('bzr.groupcompress.max_bytes_to_index')
        if val is not None:
            try:
                val = int(val)
            except ValueError as e:
                trace.warning('Value for "bzr.groupcompress.max_bytes_to_index" %r is not an integer' % (val,))
                val = None
        if val is None:
            val = self._DEFAULT_MAX_BYTES_TO_INDEX
        self._max_bytes_to_index = val
    return {'max_bytes_to_index': self._max_bytes_to_index}