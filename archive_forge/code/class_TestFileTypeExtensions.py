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
class TestFileTypeExtensions(PecanTestCase):

    @property
    def app_(self):
        """
        Test extension splits
        """

        class RootController(object):

            @expose(content_type=None)
            def _default(self, *args):
                ext = request.pecan['extension']
                assert len(args) == 1
                if ext:
                    assert ext not in args[0]
                return ext or ''
        return TestApp(Pecan(RootController()))

    def test_html_extension(self):
        for path in ('/index.html', '/index.html/'):
            r = self.app_.get(path)
            assert r.status_int == 200
            assert r.body == b'.html'

    def test_image_extension(self):
        for path in ('/index.png', '/index.png/'):
            r = self.app_.get(path)
            assert r.status_int == 200
            assert r.body == b'.png'

    def test_hidden_file(self):
        for path in ('/.vimrc', '/.vimrc/'):
            r = self.app_.get(path)
            assert r.status_int == 204
            assert r.body == b''

    def test_multi_dot_extension(self):
        for path in ('/gradient.min.js', '/gradient.min.js/'):
            r = self.app_.get(path)
            assert r.status_int == 200
            assert r.body == b'.js'

    def test_bad_content_type(self):

        class RootController(object):

            @expose()
            def index(self):
                return '/'
        app = TestApp(Pecan(RootController()))
        r = app.get('/')
        assert r.status_int == 200
        assert r.body == b'/'
        r = app.get('/index.html', expect_errors=True)
        assert r.status_int == 200
        assert r.body == b'/'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = app.get('/index.txt', expect_errors=True)
            assert r.status_int == 404

    def test_unknown_file_extension(self):

        class RootController(object):

            @expose(content_type=None)
            def _default(self, *args):
                assert 'example:x.tiny' in args
                assert request.pecan['extension'] is None
                return 'SOME VALUE'
        app = TestApp(Pecan(RootController()))
        r = app.get('/example:x.tiny')
        assert r.status_int == 200
        assert r.body == b'SOME VALUE'

    def test_guessing_disabled(self):

        class RootController(object):

            @expose(content_type=None)
            def _default(self, *args):
                assert 'index.html' in args
                assert request.pecan['extension'] is None
                return 'SOME VALUE'
        app = TestApp(Pecan(RootController(), guess_content_type_from_ext=False))
        r = app.get('/index.html')
        assert r.status_int == 200
        assert r.body == b'SOME VALUE'

    def test_content_type_guessing_disabled(self):

        class ResourceController(object):

            def __init__(self, name):
                self.name = name
                assert self.name == 'file.html'

            @expose('json')
            def index(self):
                return dict(name=self.name)

        class RootController(object):

            @expose()
            def _lookup(self, name, *remainder):
                return (ResourceController(name), remainder)
        app = TestApp(Pecan(RootController(), guess_content_type_from_ext=False))
        r = app.get('/file.html/')
        assert r.status_int == 200
        result = dict(json.loads(r.body.decode()))
        assert result == {'name': 'file.html'}
        r = app.get('/file.html')
        assert r.status_int == 302
        r = r.follow()
        result = dict(json.loads(r.body.decode()))
        assert result == {'name': 'file.html'}