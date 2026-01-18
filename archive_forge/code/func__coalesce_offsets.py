import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
@staticmethod
def _coalesce_offsets(offsets, limit=0, fudge_factor=0, max_size=0):
    """Yield coalesced offsets.

        With a long list of neighboring requests, combine them
        into a single large request, while retaining the original
        offsets.
        Turns  [(15, 10), (25, 10)] => [(15, 20, [(0, 10), (10, 10)])]
        Note that overlapping requests are not permitted. (So [(15, 10), (20,
        10)] will raise a ValueError.) This is because the data we access never
        overlaps, and it allows callers to trust that we only need any byte of
        data for 1 request (so nothing needs to be buffered to fulfill a second
        request.)

        :param offsets: A list of (start, length) pairs
        :param limit: Only combine a maximum of this many pairs Some transports
                penalize multiple reads more than others, and sometimes it is
                better to return early.
                0 means no limit
        :param fudge_factor: All transports have some level of 'it is
                better to read some more data and throw it away rather
                than seek', so collapse if we are 'close enough'
        :param max_size: Create coalesced offsets no bigger than this size.
                When a single offset is bigger than 'max_size', it will keep
                its size and be alone in the coalesced offset.
                0 means no maximum size.
        :return: return a list of _CoalescedOffset objects, which have members
            for where to start, how much to read, and how to split those chunks
            back up
        """
    last_end = None
    cur = _CoalescedOffset(None, None, [])
    coalesced_offsets = []
    if max_size <= 0:
        max_size = 100 * 1024 * 1024
    for start, size in offsets:
        end = start + size
        if last_end is not None and start <= last_end + fudge_factor and (start >= cur.start) and (limit <= 0 or len(cur.ranges) < limit) and (max_size <= 0 or end - cur.start <= max_size):
            if start < last_end:
                raise ValueError('Overlapping range not allowed: last range ended at %s, new one starts at %s' % (last_end, start))
            cur.length = end - cur.start
            cur.ranges.append((start - cur.start, size))
        else:
            if cur.start is not None:
                coalesced_offsets.append(cur)
            cur = _CoalescedOffset(start, size, [(0, size)])
        last_end = end
    if cur.start is not None:
        coalesced_offsets.append(cur)
    return coalesced_offsets