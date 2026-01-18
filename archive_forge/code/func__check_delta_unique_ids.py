from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _check_delta_unique_ids(delta):
    """Decorate a delta and check that the file ids in it are unique.

    :return: A generator over delta.
    """
    ids = set()
    for item in delta:
        length = len(ids) + 1
        ids.add(item[2])
        if len(ids) != length:
            raise errors.InconsistentDelta(item[0] or item[1], item[2], 'repeated file_id')
        yield item