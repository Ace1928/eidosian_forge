import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
class RequestBase(Middleware):
    """Test middleware, implements new model."""

    def process_response(self, response, request):
        self.called_with_request = True
        return response