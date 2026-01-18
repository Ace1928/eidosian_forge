import os
import random
import stat
import time
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from .. import config, debug, errors, osutils, ui, urlutils
from ..tests.test_server import TestServer
from ..trace import mutter
from . import (ConnectedTransport, FileExists, FileStream, NoSuchFile,
def _setmode(self, relpath, mode):
    """Set permissions on a path.

        Only set permissions on Unix systems
        """
    if 'gio' in debug.debug_flags:
        mutter('GIO _setmode %s' % relpath)
    if mode:
        try:
            f = self._get_GIO(relpath)
            f.set_attribute_uint32(gio.FILE_ATTRIBUTE_UNIX_MODE, mode)
        except gio.Error as e:
            if e.code == gio.ERROR_NOT_SUPPORTED:
                mutter('GIO Could not set permissions to %s on %s. %s', oct(mode), self._remote_path(relpath), str(e))
            else:
                self._translate_gio_error(e, relpath)