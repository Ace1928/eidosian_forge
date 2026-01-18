import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def _example_fail(self, method, url, body, headers):
    return (httplib.FORBIDDEN, 'Oh No!', {'X-Foo': 'fail'}, httplib.responses[httplib.FORBIDDEN])