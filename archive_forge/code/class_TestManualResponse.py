import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
class TestManualResponse(PecanTestCase):

    def test_manual_response(self):

        class RootController(object):

            @expose()
            def index(self, req, resp):
                resp = webob.Response(resp.environ)
                resp.body = b'Hello, World!'
                return resp
        app = TestApp(Pecan(RootController(), use_context_locals=False))
        r = app.get('/')
        assert r.body == b'Hello, World!', r.body