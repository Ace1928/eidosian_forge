import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def _prepare_body(self, original):
    """Prepares the given HTTP body data."""
    body = original.data
    if body == b'':
        body = None
    if isinstance(body, dict):
        params = [self._to_utf8(item) for item in body.items()]
        body = urlencode(params, doseq=True)
    return body