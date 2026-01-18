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
def _find_from_fallback(self, missing):
    """Find whatever keys you can from the fallbacks.

        :param missing: A set of missing keys. This set will be mutated as keys
            are found from a fallback_vfs
        :return: (parent_map, key_to_source_map, source_results)
            parent_map  the overall key => parent_keys
            key_to_source_map   a dict from {key: source}
            source_results      a list of (source: keys)
        """
    parent_map = {}
    key_to_source_map = {}
    source_results = []
    for source in self._immediate_fallback_vfs:
        if not missing:
            break
        source_parents = source.get_parent_map(missing)
        parent_map.update(source_parents)
        source_parents = list(source_parents)
        source_results.append((source, source_parents))
        key_to_source_map.update(((key, source) for key in source_parents))
        missing.difference_update(source_parents)
    return (parent_map, key_to_source_map, source_results)