import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
def _translate_path(self, path):
    """Translate a /-separated PATH to the local filename syntax.

        Note that we're translating http URLs here, not file URLs.
        The URL root location is the server's startup directory.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)

        Override from python standard library to stop it calling os.getcwd()
        """
    path = urlparse(path)[2]
    path = posixpath.normpath(urlutils.unquote(path))
    words = path.split('/')
    path = self._cwd
    for num, word in enumerate((w for w in words if w)):
        if num == 0:
            drive, word = os.path.splitdrive(word)
        head, word = os.path.split(word)
        if word in (os.curdir, os.pardir):
            continue
        path = os.path.join(path, word)
    return path