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
class TestCanonicalLookups(PecanTestCase):

    @property
    def app_(self):

        class LookupController(object):

            def __init__(self, someID):
                self.someID = someID

            @expose()
            def index(self, req, resp):
                return self.someID

        class UserController(object):

            @expose()
            def _lookup(self, someID, *remainder):
                return (LookupController(someID), remainder)

        class RootController(object):
            users = UserController()
        return TestApp(Pecan(RootController(), use_context_locals=False))

    def test_canonical_lookup(self):
        assert self.app_.get('/users', expect_errors=404).status_int == 404
        assert self.app_.get('/users/', expect_errors=404).status_int == 404
        assert self.app_.get('/users/100').status_int == 302
        assert self.app_.get('/users/100/').body == b'100'