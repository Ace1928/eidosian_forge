import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _update_pb(self, pb, msg, count, total):
    """Update the progress bar based on the current count
        and total available, total may be None if it was
        not possible to determine.
        """
    if pb is None:
        return
    if total is None:
        pb.update(msg, count, count + 1)
    else:
        pb.update(msg, count, total)