from test import SHORT_TIMEOUT
from test.with_dummyserver import test_connectionpool
import pytest
import dummyserver.testcase
import urllib3.exceptions
import urllib3.util.retry
import urllib3.util.url
from urllib3.contrib import appengine
class MockPool(object):

    def __init__(self, host, port, manager, scheme='http'):
        self.host = host
        self.port = port
        self.manager = manager
        self.scheme = scheme

    def request(self, method, url, *args, **kwargs):
        url = self._absolute_url(url)
        return self.manager.request(method, url, *args, **kwargs)

    def urlopen(self, method, url, *args, **kwargs):
        url = self._absolute_url(url)
        return self.manager.urlopen(method, url, *args, **kwargs)

    def _absolute_url(self, path):
        return urllib3.util.url.Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url