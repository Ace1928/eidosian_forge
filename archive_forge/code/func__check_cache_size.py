from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def _check_cache_size(self, inv_size, entry_cache):
    """Check that the entry_cache is large enough.

        We want the cache to be ~2x the size of an inventory. The reason is
        because we use a FIFO cache, and how Inventory records are likely to
        change. In general, you have a small number of records which change
        often, and a lot of records which do not change at all. So when the
        cache gets full, you actually flush out a lot of the records you are
        interested in, which means you need to recreate all of those records.
        An LRU Cache would be better, but the overhead negates the cache
        coherency benefit.

        One way to look at it, only the size of the cache > len(inv) is your
        'working' set. And in general, it shouldn't be a problem to hold 2
        inventories in memory anyway.

        :param inv_size: The number of entries in an inventory.
        """
    if entry_cache is None:
        return
    recommended_min_cache_size = inv_size * 1.5
    if entry_cache.cache_size() < recommended_min_cache_size:
        recommended_cache_size = inv_size * 2
        trace.mutter('Resizing the inventory entry cache from %d to %d', entry_cache.cache_size(), recommended_cache_size)
        entry_cache.resize(recommended_cache_size)