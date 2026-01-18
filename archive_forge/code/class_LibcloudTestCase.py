import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
class LibcloudTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self._visited_urls = []
        self._executed_mock_methods = []
        super().__init__(*args, **kwargs)

    def setUp(self):
        self._visited_urls = []
        self._executed_mock_methods = []

    def _add_visited_url(self, url):
        self._visited_urls.append(url)

    def _add_executed_mock_method(self, method_name):
        self._executed_mock_methods.append(method_name)

    def assertExecutedMethodCount(self, expected):
        actual = len(self._executed_mock_methods)
        self.assertEqual(actual, expected, 'expected %d, but %d mock methods were executed' % (expected, actual))