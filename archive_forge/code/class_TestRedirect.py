import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
class TestRedirect(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def index(self):
                redirect('/testing')

            @expose()
            def internal(self):
                redirect('/testing', internal=True)

            @expose()
            def bad_internal(self):
                redirect('/testing', internal=True, code=301)

            @expose()
            def permanent(self):
                redirect('/testing', code=301)

            @expose()
            def testing(self):
                return 'it worked!'
        return TestApp(make_app(RootController(), debug=False))

    def test_index(self):
        r = self.app_.get('/')
        assert r.status_int == 302
        r = r.follow()
        assert r.status_int == 200
        assert r.body == b'it worked!'

    def test_internal(self):
        r = self.app_.get('/internal')
        assert r.status_int == 200
        assert r.body == b'it worked!'

    def test_internal_with_301(self):
        self.assertRaises(ValueError, self.app_.get, '/bad_internal')

    def test_permanent_redirect(self):
        r = self.app_.get('/permanent')
        assert r.status_int == 301
        r = r.follow()
        assert r.status_int == 200
        assert r.body == b'it worked!'

    def test_x_forward_proto(self):

        class ChildController(object):

            @expose()
            def index(self):
                redirect('/testing')

        class RootController(object):

            @expose()
            def index(self):
                redirect('/testing')

            @expose()
            def testing(self):
                return 'it worked!'
            child = ChildController()
        app = TestApp(make_app(RootController(), debug=True))
        res = app.get('/child', extra_environ=dict(HTTP_X_FORWARDED_PROTO='https'))
        assert res.status_int == 302
        assert res.location == 'https://localhost/child/'
        assert res.request.environ['HTTP_X_FORWARDED_PROTO'] == 'https'