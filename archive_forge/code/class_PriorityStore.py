from __future__ import annotations
from heapq import heappop, heappush
from typing import (
from simpy.core import BoundClass, Environment
from simpy.resources import base
class PriorityStore(Store):
    """Resource with *capacity* slots for storing objects in priority order.

    Unlike :class:`Store` which provides first-in first-out discipline,
    :class:`PriorityStore` maintains items in sorted order such that
    the smallest items value are retrieved first from the store.

    All items in a *PriorityStore* instance must be order-able; which is to say
    that items must implement :meth:`~object.__lt__()`. To use unorderable
    items with *PriorityStore*, use :class:`PriorityItem`.

    """

    def _do_put(self, event: StorePut) -> Optional[bool]:
        if len(self.items) < self._capacity:
            heappush(self.items, event.item)
            event.succeed()
        return None

    def _do_get(self, event: StoreGet) -> Optional[bool]:
        if self.items:
            event.succeed(heappop(self.items))
        return None