import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
class TestHTTPClientMixin(object, metaclass=abc.ABCMeta):

    def setUp(self):
        super(TestHTTPClientMixin, self).setUp()
        self.requests = self.useFixture(mock_fixture.Fixture())
        self.http = self.initialize()

    @abc.abstractmethod
    def initialize(self):
        """Return client class, instance."""

    def _test_headers(self, expected_headers, **kwargs):
        self.requests.register_uri(METHOD, URL, request_headers=expected_headers)
        self.http.request(URL, METHOD, **kwargs)
        self.assertEqual(kwargs.get('body'), self.requests.last_request.body)

    def test_headers_without_body(self):
        self._test_headers({'Accept': 'application/json'})

    def test_headers_with_body(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        self._test_headers(headers, body=BODY)

    def test_headers_without_body_with_content_type(self):
        headers = {'Accept': 'application/json'}
        self._test_headers(headers, content_type='application/json')

    def test_headers_with_body_with_content_type(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        self._test_headers(headers, body=BODY, content_type='application/json')

    def test_headers_defined_in_headers(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        self._test_headers(headers, body=BODY, headers=headers)

    def test_osprofiler_headers_are_injected(self):
        osprofiler.profiler.init('SWORDFISH')
        self.addCleanup(osprofiler.profiler.clean)
        headers = {'Accept': 'application/json'}
        headers.update(osprofiler.web.get_trace_id_headers())
        self._test_headers(headers)