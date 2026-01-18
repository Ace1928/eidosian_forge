from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _check_delta_ids_match_entry(delta):
    """Decorate a delta and check that the ids in it match the entry.file_id.

    :return: A generator over delta.
    """
    for item in delta:
        entry = item[3]
        if entry is not None:
            if entry.file_id != item[2]:
                raise errors.InconsistentDelta(item[0] or item[1], item[2], 'mismatched id with %r' % entry)
        yield item