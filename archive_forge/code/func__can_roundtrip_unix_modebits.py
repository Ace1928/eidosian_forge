import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _can_roundtrip_unix_modebits(self):
    """Return true if this transport can store and retrieve unix modebits.

        (For example, 0700 to make a directory owner-private.)

        Note: most callers will not want to switch on this, but should rather
        just try and set permissions and let them be either stored or not.
        This is intended mainly for the use of the test suite.

        Warning: this is not guaranteed to be accurate as sometimes we can't
        be sure: for example with vfat mounted on unix, or a windows sftp
        server."""
    return False