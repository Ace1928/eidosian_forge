import atexit
import errno
import os
import re
import shutil
import sys
import tempfile
from hashlib import md5
from io import BytesIO
from json import dumps
from time import sleep
from httplib2 import Http, urlnorm
from wadllib.application import Application
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError, error_for
from lazr.uri import URI
def _getCachedHeader(self, uri, header):
    """Retrieve a cached value for an HTTP header."""
    scheme, authority, request_uri, cachekey = urlnorm(uri)
    cached_value = self.get(cachekey)
    header_start = header + ':'
    if not isinstance(header_start, bytes):
        header_start = header_start.encode('utf-8')
    if cached_value is not None:
        for line in BytesIO(cached_value):
            if line.startswith(header_start):
                return line[len(header_start):].strip()
    return None