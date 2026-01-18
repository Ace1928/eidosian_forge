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
class TestEmptyContent(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def index(self):
                pass

            @expose()
            def explicit_body(self):
                response.body = b'Hello, World!'

            @expose()
            def empty_body(self):
                response.body = b''

            @expose()
            def explicit_text(self):
                response.text = 'Hello, World!'

            @expose()
            def empty_text(self):
                response.text = ''

            @expose()
            def explicit_json(self):
                response.json = {'foo': 'bar'}

            @expose()
            def explicit_json_body(self):
                response.json_body = {'foo': 'bar'}

            @expose()
            def non_unicode(self):
                return chr(192)
        return TestApp(Pecan(RootController()))

    def test_empty_index(self):
        r = self.app_.get('/')
        self.assertEqual(r.status_int, 204)
        self.assertNotIn('Content-Type', r.headers)
        self.assertEqual(r.headers['Content-Length'], '0')
        self.assertEqual(len(r.body), 0)

    def test_index_with_non_unicode(self):
        r = self.app_.get('/non_unicode/')
        self.assertEqual(r.status_int, 200)

    def test_explicit_body(self):
        r = self.app_.get('/explicit_body/')
        self.assertEqual(r.status_int, 200)
        self.assertEqual(r.body, b'Hello, World!')

    def test_empty_body(self):
        r = self.app_.get('/empty_body/')
        self.assertEqual(r.status_int, 204)
        self.assertEqual(r.body, b'')

    def test_explicit_text(self):
        r = self.app_.get('/explicit_text/')
        self.assertEqual(r.status_int, 200)
        self.assertEqual(r.body, b'Hello, World!')

    def test_empty_text(self):
        r = self.app_.get('/empty_text/')
        self.assertEqual(r.status_int, 204)
        self.assertEqual(r.body, b'')

    def test_explicit_json(self):
        r = self.app_.get('/explicit_json/')
        self.assertEqual(r.status_int, 200)
        json_resp = json.loads(r.body.decode())
        assert json_resp == {'foo': 'bar'}

    def test_explicit_json_body(self):
        r = self.app_.get('/explicit_json_body/')
        self.assertEqual(r.status_int, 200)
        json_resp = json.loads(r.body.decode())
        assert json_resp == {'foo': 'bar'}