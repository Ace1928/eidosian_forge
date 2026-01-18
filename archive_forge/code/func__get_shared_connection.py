import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _get_shared_connection(self):
    """Get the object shared amongst cloned transports.

        This should be used only by classes that needs to extend the sharing
        with objects other than transports.

        Use _get_connection to get the connection itself.
        """
    return self._shared_connection