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
class TestDispatch(PecanTestCase):

    @property
    def app_(self):

        class SubSubController(object):

            @expose()
            def index(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return '/sub/sub/'

            @expose()
            def deeper(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return '/sub/sub/deeper'

        class SubController(object):

            @expose()
            def index(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return '/sub/'

            @expose()
            def deeper(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return '/sub/deeper'
            sub = SubSubController()

        class RootController(object):

            @expose()
            def index(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return '/'

            @expose()
            def deeper(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return '/deeper'
            sub = SubController()
        return TestApp(Pecan(RootController(), use_context_locals=False))

    def test_index(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert r.body == b'/'

    def test_one_level(self):
        r = self.app_.get('/deeper')
        assert r.status_int == 200
        assert r.body == b'/deeper'

    def test_one_level_with_trailing(self):
        r = self.app_.get('/sub/')
        assert r.status_int == 200
        assert r.body == b'/sub/'

    def test_two_levels(self):
        r = self.app_.get('/sub/deeper')
        assert r.status_int == 200
        assert r.body == b'/sub/deeper'

    def test_two_levels_with_trailing(self):
        r = self.app_.get('/sub/sub/')
        assert r.status_int == 200

    def test_three_levels(self):
        r = self.app_.get('/sub/sub/deeper')
        assert r.status_int == 200
        assert r.body == b'/sub/sub/deeper'