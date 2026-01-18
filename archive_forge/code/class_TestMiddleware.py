import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
class TestMiddleware(Middleware):

    @staticmethod
    def process_request(req):
        return 'foobar'