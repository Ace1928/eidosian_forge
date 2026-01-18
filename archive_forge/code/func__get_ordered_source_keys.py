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
def _get_ordered_source_keys(self, ordering, parent_map, key_to_source_map):
    """Get the (source, [keys]) list.

        The returned objects should be in the order defined by 'ordering',
        which can weave between different sources.

        :param ordering: Must be one of 'topological' or 'groupcompress'
        :return: List of [(source, [keys])] tuples, such that all keys are in
            the defined order, regardless of source.
        """
    if ordering == 'topological':
        present_keys = tsort.topo_sort(parent_map)
    else:
        present_keys = sort_gc_optimal(parent_map)
    source_keys = []
    current_source = None
    for key in present_keys:
        source = key_to_source_map.get(key, self)
        if source is not current_source:
            source_keys.append((source, []))
            current_source = source
        source_keys[-1][1].append(key)
    return source_keys