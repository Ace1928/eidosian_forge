from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
def _test_wsgi_middleware_with_invalid_trace(self, headers, hmac_key, mock_profiler_init, enabled=True):
    request = mock.MagicMock()
    request.get_response.return_value = 'yeah!'
    request.headers = headers
    middleware = web.WsgiMiddleware('app', hmac_key, enabled=enabled)
    self.assertEqual('yeah!', middleware(request))
    request.get_response.assert_called_once_with('app')
    self.assertEqual(0, mock_profiler_init.call_count)