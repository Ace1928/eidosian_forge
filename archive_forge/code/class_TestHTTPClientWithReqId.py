import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
class TestHTTPClientWithReqId(TestHTTPClientMixin, testtools.TestCase):
    """Tests for when global_request_id is set."""

    def initialize(self):
        self.req_id = 'req-%s' % uuidutils.generate_uuid()
        return client.HTTPClient(token=AUTH_TOKEN, endpoint_url=END_URL, global_request_id=self.req_id)

    def test_request_success(self):
        headers = {'Accept': 'application/json', 'X-OpenStack-Request-ID': self.req_id}
        self.requests.register_uri(METHOD, URL, request_headers=headers)
        self.http.request(URL, METHOD)