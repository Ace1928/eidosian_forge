from __future__ import absolute_import
from .. import (
from ..helpers import (
import stat
def _iterable_as_config_list(s):
    """Format an iterable as a sequence of comma-separated strings.

    To match what ConfigObj expects, a single item list has a trailing comma.
    """
    items = sorted(s)
    if len(items) == 1:
        return '%s,' % (items[0],)
    else:
        return ', '.join(items)