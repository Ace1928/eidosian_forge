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
class GioLocalURLServer(TestServer):
    """A pretend server for local transports, using file:// urls.

    Of course no actual server is required to access the local filesystem, so
    this just exists to tell the test code how to get to it.
    """

    def start_server(self):
        pass

    def get_url(self):
        """See Transport.Server.get_url."""
        return 'gio+' + urlutils.local_path_to_url('')