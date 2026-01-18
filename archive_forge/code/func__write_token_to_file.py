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
def _write_token_to_file(self):
    """
        Write token to credential file.
        Mocked in libcloud.test.common.google.GoogleTestCase.
        """
    filename = os.path.expanduser(self.credential_file)
    filename = os.path.realpath(filename)
    try:
        data = json.dumps(self.token)
        write_flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
        with os.fdopen(os.open(filename, write_flags, int('600', 8)), 'w') as f:
            f.write(data)
    except Exception as e:
        LOG.info('Failed to write auth token to file "%s": %s', filename, str(e))