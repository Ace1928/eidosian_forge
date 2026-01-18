from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

    Requests data from cache.

    :param cache_id: ID of cache to get entries from.
    :param skip_count: *(Optional)* Number of records to skip.
    :param page_size: *(Optional)* Number of records to fetch.
    :param path_filter: *(Optional)* If present, only return the entries containing this substring in the path
    :returns: A tuple with the following items:

        0. **cacheDataEntries** - Array of object store data entries.
        1. **returnCount** - Count of returned entries from this storage. If pathFilter is empty, it is the count of all entries from this storage.
    