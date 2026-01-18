import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _crud_test(self, url):
    self.assert_head_and_get_return_same_response(url, expected_status=http.client.NOT_FOUND)
    self.put(url)
    self.assert_head_and_get_return_same_response(url, expected_status=http.client.NO_CONTENT)
    self.delete(url)
    self.assert_head_and_get_return_same_response(url, expected_status=http.client.NOT_FOUND)