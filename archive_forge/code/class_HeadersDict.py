import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
class HeadersDict(MutableMapping):
    """A case-insenseitive dictionary to represent HTTP headers."""

    def __init__(self, *args, **kwargs):
        self._dict = {}
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self._dict[_HeaderKey(key)] = value

    def __getitem__(self, key):
        return self._dict[_HeaderKey(key)]

    def __delitem__(self, key):
        del self._dict[_HeaderKey(key)]

    def __iter__(self):
        return (str(key) for key in self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return repr(self._dict)

    def copy(self):
        return HeadersDict(self.items())