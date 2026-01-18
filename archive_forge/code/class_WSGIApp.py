from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
class WSGIApp(object):

    def __init__(self, app):
        self.app = app

    def __call__(self, e, start_response):

        def keep_status_start_response(status, headers, exc_info=None):
            return start_response('404 Not Found', headers, exc_info)
        return self.app(e, keep_status_start_response)