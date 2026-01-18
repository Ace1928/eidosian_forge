import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
@staticmethod
def _is_gce():
    """
        Checks if we can access the GCE metadata server.
        Mocked in libcloud.test.common.google.GoogleTestCase.
        """
    http_code, http_reason, body = _get_gce_metadata(retry_failed=False)
    if http_code == httplib.OK and body:
        return True
    return False