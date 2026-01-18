from __future__ import annotations
from heapq import heappop, heappush
from typing import (
from simpy.core import BoundClass, Environment
from simpy.resources import base
class FilterStoreGet(StoreGet):
    """Request to get an *item* from the *store* matching the *filter*. The
    request is triggered once there is such an item available in the store.

    *filter* is a function receiving one item. It should return ``True`` for
    items matching the filter criterion. The default function returns ``True``
    for all items, which makes the request to behave exactly like
    :class:`StoreGet`.

    """

    def __init__(self, resource: FilterStore, filter: Callable[[Any], bool]=lambda item: True):
        self.filter = filter
        'The filter function to filter items in the store.'
        super().__init__(resource)