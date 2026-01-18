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
def _relpath_to_url(self, relpath):
    full_url = urlutils.join(self.url, relpath)
    if isinstance(full_url, str):
        raise urlutils.InvalidURL(full_url)
    return full_url