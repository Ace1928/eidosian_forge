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
def _mount_done_cb(self, obj, res):
    try:
        obj.mount_enclosing_volume_finish(res)
        self.loop.quit()
    except gio.Error as e:
        self.loop.quit()
        raise errors.BzrError('Failed to mount the given location: ' + str(e))