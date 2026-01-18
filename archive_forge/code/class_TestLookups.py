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
class TestLookups(PecanTestCase):

    @property
    def app_(self):

        class LookupController(object):

            def __init__(self, someID):
                self.someID = someID

            @expose()
            def index(self, req, resp):
                return '/%s' % self.someID

            @expose()
            def name(self, req, resp):
                return '/%s/name' % self.someID

        class RootController(object):

            @expose()
            def index(self, req, resp):
                return '/'

            @expose()
            def _lookup(self, someID, *remainder):
                return (LookupController(someID), remainder)
        return TestApp(Pecan(RootController(), use_context_locals=False))

    def test_index(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert r.body == b'/'

    def test_lookup(self):
        r = self.app_.get('/100/')
        assert r.status_int == 200
        assert r.body == b'/100'

    def test_lookup_with_method(self):
        r = self.app_.get('/100/name')
        assert r.status_int == 200
        assert r.body == b'/100/name'

    def test_lookup_with_wrong_argspec(self):

        class RootController(object):

            @expose()
            def _lookup(self, someID):
                return 'Bad arg spec'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app = TestApp(Pecan(RootController(), use_context_locals=False))
            r = app.get('/foo/bar', expect_errors=True)
            assert r.status_int == 404