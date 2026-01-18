from test import SHORT_TIMEOUT
from test.with_dummyserver import test_connectionpool
import pytest
import dummyserver.testcase
import urllib3.exceptions
import urllib3.util.retry
import urllib3.util.url
from urllib3.contrib import appengine
@pytest.mark.usefixtures('testbed')
class TestGAERetry(test_connectionpool.TestRetry):

    def setup_method(self, method):
        self.manager = appengine.AppEngineManager()
        self.pool = MockPool(self.host, self.port, self.manager)

    def test_default_method_whitelist_retried(self):
        """urllib3 should retry methods in the default method whitelist"""
        retry = urllib3.util.retry.Retry(total=1, status_forcelist=[418])
        resp = self.pool.request('HEAD', '/successful_retry', headers={'test-name': 'test_default_whitelist'}, retries=retry)
        assert resp.status == 200

    def test_retry_return_in_response(self):
        headers = {'test-name': 'test_retry_return_in_response'}
        retry = urllib3.util.retry.Retry(total=2, status_forcelist=[418])
        resp = self.pool.request('GET', '/successful_retry', headers=headers, retries=retry)
        assert resp.status == 200
        assert resp.retries.total == 1
        assert resp.retries.history == (urllib3.util.retry.RequestHistory('GET', self.pool._absolute_url('/successful_retry'), None, 418, None),)
    test_retry_redirect_history = None
    test_multi_redirect_history = None