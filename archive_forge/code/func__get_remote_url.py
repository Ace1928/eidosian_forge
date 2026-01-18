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
def _get_remote_url(self, path):
    path_parts = path.split(os.path.sep)
    if os.path.isabs(path):
        if path_parts[:len(self._local_path_parts)] != self._local_path_parts:
            raise BadWebserverPath(path, self.test_dir)
        remote_path = '/'.join(path_parts[len(self._local_path_parts):])
    else:
        remote_path = '/'.join(path_parts)
    return self._http_base_url + remote_path