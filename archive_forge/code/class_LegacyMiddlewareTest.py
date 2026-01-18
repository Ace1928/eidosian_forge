import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
class LegacyMiddlewareTest(Middleware):
    pass