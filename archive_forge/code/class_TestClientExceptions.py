import http.client as http_client
import eventlet.patcher
import httplib2
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
class TestClientExceptions(functional.FunctionalTest):

    def setUp(self):
        super(TestClientExceptions, self).setUp()
        self.port = utils.get_unused_port()
        server = wsgi.Server()
        self.config(bind_host='127.0.0.1')
        self.config(workers=0)
        server.start(ExceptionTestApp(), self.port)
        self.client = client.BaseClient('127.0.0.1', self.port)

    def _do_test_exception(self, path, exc_type):
        try:
            self.client.do_request('GET', path)
            self.fail('expected %s' % exc_type)
        except exc_type as e:
            if 'retry' in path:
                self.assertEqual(10, e.retry_after)

    def test_rate_limited(self):
        """
        Test rate limited response
        """
        self._do_test_exception('/rate-limit', exception.LimitExceeded)

    def test_rate_limited_retry(self):
        """
        Test rate limited response with retry
        """
        self._do_test_exception('/rate-limit-retry', exception.LimitExceeded)

    def test_service_unavailable(self):
        """
        Test service unavailable response
        """
        self._do_test_exception('/service-unavailable', exception.ServiceUnavailable)

    def test_service_unavailable_retry(self):
        """
        Test service unavailable response with retry
        """
        self._do_test_exception('/service-unavailable-retry', exception.ServiceUnavailable)

    def test_expectation_failed(self):
        """
        Test expectation failed response
        """
        self._do_test_exception('/expectation-failed', exception.UnexpectedStatus)

    def test_server_error(self):
        """
        Test server error response
        """
        self._do_test_exception('/server-error', exception.ServerError)

    def test_server_traceback(self):
        """
        Verify that the wsgi server does not return tracebacks to the client on
        500 errors (bug 1192132)
        """
        http = httplib2.Http()
        path = 'http://%s:%d/server-traceback' % ('127.0.0.1', self.port)
        response, content = http.request(path, 'GET')
        self.assertNotIn(b'ServerError', content)
        self.assertEqual(http_client.INTERNAL_SERVER_ERROR, response.status)