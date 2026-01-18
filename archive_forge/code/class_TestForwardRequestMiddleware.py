from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
class TestForwardRequestMiddleware(Middleware):

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'] != '/not_found':
            return self.app(environ, start_response)
        environ['PATH_INFO'] = self.url

        def factory(app):

            class WSGIApp(object):

                def __init__(self, app):
                    self.app = app

                def __call__(self, e, start_response):

                    def keep_status_start_response(status, headers, exc_info=None):
                        return start_response('404 Not Found', headers, exc_info)
                    return self.app(e, keep_status_start_response)
            return WSGIApp(app)
        raise ForwardRequestException(factory=factory)