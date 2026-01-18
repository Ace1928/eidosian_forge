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
class TestEngines(PecanTestCase):
    template_path = os.path.join(os.path.dirname(__file__), 'templates')

    @unittest.skipIf('genshi' not in builtin_renderers, 'Genshi not installed')
    def test_genshi(self):

        class RootController(object):

            @expose('genshi:genshi.html')
            def index(self, name='Jonathan'):
                return dict(name=name)

            @expose('genshi:genshi_bad.html')
            def badtemplate(self):
                return dict()
        app = TestApp(Pecan(RootController(), template_path=self.template_path))
        r = app.get('/')
        assert r.status_int == 200
        assert b'<h1>Hello, Jonathan!</h1>' in r.body
        r = app.get('/index.html?name=World')
        assert r.status_int == 200
        assert b'<h1>Hello, World!</h1>' in r.body
        error_msg = None
        try:
            r = app.get('/badtemplate.html')
        except Exception as e:
            for error_f in error_formatters:
                error_msg = error_f(e)
                if error_msg:
                    break
        assert error_msg is not None

    @unittest.skipIf('kajiki' not in builtin_renderers, 'Kajiki not installed')
    def test_kajiki(self):

        class RootController(object):

            @expose('kajiki:kajiki.html')
            def index(self, name='Jonathan'):
                return dict(name=name)
        app = TestApp(Pecan(RootController(), template_path=self.template_path))
        r = app.get('/')
        assert r.status_int == 200
        assert b'<h1>Hello, Jonathan!</h1>' in r.body
        r = app.get('/index.html?name=World')
        assert r.status_int == 200
        assert b'<h1>Hello, World!</h1>' in r.body

    @unittest.skipIf('jinja' not in builtin_renderers, 'Jinja not installed')
    def test_jinja(self):

        class RootController(object):

            @expose('jinja:jinja.html')
            def index(self, name='Jonathan'):
                return dict(name=name)

            @expose('jinja:jinja_bad.html')
            def badtemplate(self):
                return dict()
        app = TestApp(Pecan(RootController(), template_path=self.template_path))
        r = app.get('/')
        assert r.status_int == 200
        assert b'<h1>Hello, Jonathan!</h1>' in r.body
        error_msg = None
        try:
            r = app.get('/badtemplate.html')
        except Exception as e:
            for error_f in error_formatters:
                error_msg = error_f(e)
                if error_msg:
                    break
        assert error_msg is not None

    @unittest.skipIf('mako' not in builtin_renderers, 'Mako not installed')
    def test_mako(self):

        class RootController(object):

            @expose('mako:mako.html')
            def index(self, name='Jonathan'):
                return dict(name=name)

            @expose('mako:mako_bad.html')
            def badtemplate(self):
                return dict()
        app = TestApp(Pecan(RootController(), template_path=self.template_path))
        r = app.get('/')
        assert r.status_int == 200
        assert b'<h1>Hello, Jonathan!</h1>' in r.body
        r = app.get('/index.html?name=World')
        assert r.status_int == 200
        assert b'<h1>Hello, World!</h1>' in r.body
        error_msg = None
        try:
            r = app.get('/badtemplate.html')
        except Exception as e:
            for error_f in error_formatters:
                error_msg = error_f(e)
                if error_msg:
                    break
        assert error_msg is not None

    def test_renderer_not_found(self):

        class RootController(object):

            @expose('mako3:mako.html')
            def index(self, name='Jonathan'):
                return dict(name=name)
        app = TestApp(Pecan(RootController(), template_path=self.template_path))
        try:
            r = app.get('/')
        except Exception as e:
            expected = e
        assert 'support for "mako3" was not found;' in str(expected)

    def test_json(self):
        expected_result = dict(name='Jonathan', age=30, nested=dict(works=True))

        class RootController(object):

            @expose('json')
            def index(self):
                return expected_result
        app = TestApp(Pecan(RootController()))
        r = app.get('/')
        assert r.status_int == 200
        result = json.loads(r.body.decode())
        assert result == expected_result

    def test_custom_renderer(self):

        class RootController(object):

            @expose('backwards:mako.html')
            def index(self, name='Joe'):
                return dict(name=name)

        class BackwardsRenderer(MakoRenderer):

            def render(self, template_path, namespace):
                namespace = dict(((k, v[::-1]) for k, v in namespace.items()))
                return super(BackwardsRenderer, self).render(template_path, namespace)
        app = TestApp(Pecan(RootController(), template_path=self.template_path, custom_renderers={'backwards': BackwardsRenderer}))
        r = app.get('/')
        assert r.status_int == 200
        assert b'<h1>Hello, eoJ!</h1>' in r.body
        r = app.get('/index.html?name=Tim')
        assert r.status_int == 200
        assert b'<h1>Hello, miT!</h1>' in r.body

    def test_override_template(self):

        class RootController(object):

            @expose('foo.html')
            def index(self):
                override_template(None, content_type='text/plain')
                return 'Override'
        app = TestApp(Pecan(RootController()))
        r = app.get('/')
        assert r.status_int == 200
        assert b'Override' in r.body
        assert r.content_type == 'text/plain'

    def test_render(self):

        class RootController(object):

            @expose()
            def index(self, name='Jonathan'):
                return render('mako.html', dict(name=name))
        app = TestApp(Pecan(RootController(), template_path=self.template_path))
        r = app.get('/')
        assert r.status_int == 200
        assert b'<h1>Hello, Jonathan!</h1>' in r.body

    def test_default_json_renderer(self):

        class RootController(object):

            @expose()
            def index(self, name='Bill'):
                return dict(name=name)
        app = TestApp(Pecan(RootController(), default_renderer='json'))
        r = app.get('/')
        assert r.status_int == 200
        result = dict(json.loads(r.body.decode()))
        assert result == {'name': 'Bill'}

    def test_default_json_renderer_with_explicit_content_type(self):

        class RootController(object):

            @expose(content_type='text/plain')
            def index(self, name='Bill'):
                return name
        app = TestApp(Pecan(RootController(), default_renderer='json'))
        r = app.get('/')
        assert r.status_int == 200
        assert r.body == b'Bill'