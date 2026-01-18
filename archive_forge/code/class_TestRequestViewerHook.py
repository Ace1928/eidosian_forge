import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
class TestRequestViewerHook(PecanTestCase):

    def test_basic_single_default_hook(self):
        _stdout = StringIO()

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'
        app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(writer=_stdout)]))
        response = app.get('/')
        out = _stdout.getvalue()
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert 'path' in out
        assert 'method' in out
        assert 'status' in out
        assert 'method' in out
        assert 'params' in out
        assert 'hooks' in out
        assert '200 OK' in out
        assert "['RequestViewerHook']" in out
        assert '/' in out

    def test_bad_response_from_app(self):
        """When exceptions are raised the hook deals with them properly"""
        _stdout = StringIO()

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'
        app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(writer=_stdout)]))
        response = app.get('/404', expect_errors=True)
        out = _stdout.getvalue()
        assert response.status_int == 404
        assert 'path' in out
        assert 'method' in out
        assert 'status' in out
        assert 'method' in out
        assert 'params' in out
        assert 'hooks' in out
        assert '404 Not Found' in out
        assert "['RequestViewerHook']" in out
        assert '/' in out

    def test_single_item(self):
        _stdout = StringIO()

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'
        app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(config={'items': ['path']}, writer=_stdout)]))
        response = app.get('/')
        out = _stdout.getvalue()
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert '/' in out
        assert 'path' in out
        assert 'method' not in out
        assert 'status' not in out
        assert 'method' not in out
        assert 'params' not in out
        assert 'hooks' not in out
        assert '200 OK' not in out
        assert "['RequestViewerHook']" not in out

    def test_single_blacklist_item(self):
        _stdout = StringIO()

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'
        app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(config={'blacklist': ['/']}, writer=_stdout)]))
        response = app.get('/')
        out = _stdout.getvalue()
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert out == ''

    def test_item_not_in_defaults(self):
        _stdout = StringIO()

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'
        app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(config={'items': ['date']}, writer=_stdout)]))
        response = app.get('/')
        out = _stdout.getvalue()
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert 'date' in out
        assert 'method' not in out
        assert 'status' not in out
        assert 'method' not in out
        assert 'params' not in out
        assert 'hooks' not in out
        assert '200 OK' not in out
        assert "['RequestViewerHook']" not in out
        assert '/' not in out

    def test_hook_formatting(self):
        hooks = ['<pecan.hooks.RequestViewerHook object at 0x103a5f910>']
        viewer = RequestViewerHook()
        formatted = viewer.format_hooks(hooks)
        assert formatted == ['RequestViewerHook']

    def test_deal_with_pecan_configs(self):
        """If config comes from pecan.conf convert it to dict"""
        conf = Config(conf_dict={'items': ['url']})
        viewer = RequestViewerHook(conf)
        assert viewer.items == ['url']