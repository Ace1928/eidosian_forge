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
class ExceptionTestApp(object):
    """
    Test WSGI application which can respond with multiple kinds of HTTP
    status codes
    """

    @webob.dec.wsgify
    def __call__(self, request):
        path = request.path_qs
        if path == '/rate-limit':
            request.response = webob.exc.HTTPRequestEntityTooLarge()
        elif path == '/rate-limit-retry':
            request.response.retry_after = 10
            request.response.status = http_client.REQUEST_ENTITY_TOO_LARGE
        elif path == '/service-unavailable':
            request.response = webob.exc.HTTPServiceUnavailable()
        elif path == '/service-unavailable-retry':
            request.response.retry_after = 10
            request.response.status = http_client.SERVICE_UNAVAILABLE
        elif path == '/expectation-failed':
            request.response = webob.exc.HTTPExpectationFailed()
        elif path == '/server-error':
            request.response = webob.exc.HTTPServerError()
        elif path == '/server-traceback':
            raise exception.ServerError()