import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def _add_visited_url(self, url):
    self._visited_urls.append(url)