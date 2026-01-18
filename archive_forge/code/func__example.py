import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def _example(self, method, url, body, headers):
    """
        Return a simple message and header, regardless of input.
        """
    return (httplib.OK, 'Hello World!', {'X-Foo': 'libcloud'}, httplib.responses[httplib.OK])