from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _check_delta_ids_are_valid(delta):
    """Decorate a delta and check that the ids in it are valid.

    :return: A generator over delta.
    """
    for item in delta:
        entry = item[3]
        if item[2] is None:
            raise errors.InconsistentDelta(item[0] or item[1], item[2], 'entry with file_id None %r' % entry)
        if not isinstance(item[2], bytes):
            raise errors.InconsistentDelta(item[0] or item[1], item[2], 'entry with non bytes file_id %r' % entry)
        yield item