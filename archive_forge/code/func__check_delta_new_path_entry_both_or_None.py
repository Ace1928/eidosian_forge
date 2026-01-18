from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _check_delta_new_path_entry_both_or_None(delta):
    """Decorate a delta and check that the new_path and entry are paired.

    :return: A generator over delta.
    """
    for item in delta:
        new_path = item[1]
        entry = item[3]
        if new_path is None and entry is not None:
            raise errors.InconsistentDelta(item[0], item[1], 'Entry with no new_path')
        if new_path is not None and entry is None:
            raise errors.InconsistentDelta(new_path, item[1], 'new_path with no entry')
        yield item