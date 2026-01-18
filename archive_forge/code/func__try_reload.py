import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _try_reload(self, error):
    """We just got a NoSuchFile exception.

        Try to reload the indices, if it fails, just raise the current
        exception.
        """
    if self._reload_func is None:
        return False
    trace.mutter('Trying to reload after getting exception: %s', str(error))
    if not self._reload_func():
        trace.mutter('_reload_func indicated nothing has changed. Raising original exception.')
        return False
    return True