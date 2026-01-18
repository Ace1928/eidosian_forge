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
class GioStatResult:

    def __init__(self, f):
        info = f.query_info('standard::size,standard::type')
        self.st_size = info.get_size()
        type = info.get_file_type()
        if type == gio.FILE_TYPE_REGULAR:
            self.st_mode = stat.S_IFREG
        elif type == gio.FILE_TYPE_DIRECTORY:
            self.st_mode = stat.S_IFDIR