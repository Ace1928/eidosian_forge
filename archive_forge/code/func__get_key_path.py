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
def _get_key_path(self, key):
    """Return the path on disk where ``key`` is stored."""
    safe_key = self._get_safe_name(key)
    if safe_key.startswith(self.TEMPFILE_PREFIX):
        raise ValueError("Cache key cannot start with '%s'" % self.TEMPFILE_PREFIX)
    return os.path.join(self._cache_dir, safe_key)