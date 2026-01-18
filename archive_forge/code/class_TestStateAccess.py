import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
class TestStateAccess(PecanTestCase):

    def setUp(self):
        super(TestStateAccess, self).setUp()
        self.args = None

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'

            @expose()
            def greet(self, name):
                return 'Hello, %s!' % name

            @expose()
            def greetmore(self, *args):
                return 'Hello, %s!' % args[0]

            @expose()
            def kwargs(self, **kw):
                return 'Hello, %s!' % kw['name']

            @expose()
            def mixed(self, first, second, *args):
                return 'Mixed'

        class SimpleHook(PecanHook):

            def before(inself, state):
                self.args = (state.controller, state.arguments)
        self.root = RootController()
        self.app = TestApp(make_app(self.root, hooks=[SimpleHook()]))

    def test_no_args(self):
        self.app.get('/')
        assert self.args[0] == self.root.index
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_single_arg(self):
        self.app.get('/greet/joe')
        assert self.args[0] == self.root.greet
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['joe']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_single_vararg(self):
        self.app.get('/greetmore/joe')
        assert self.args[0] == self.root.greetmore
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == ['joe']
        assert kwargs(self.args[1]) == {}

    def test_single_kw(self):
        self.app.get('/kwargs/?name=joe')
        assert self.args[0] == self.root.kwargs
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'name': 'joe'}

    def test_single_kw_post(self):
        self.app.post('/kwargs/', params={'name': 'joe'})
        assert self.args[0] == self.root.kwargs
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'name': 'joe'}

    def test_mixed_args(self):
        self.app.get('/mixed/foo/bar/spam/eggs')
        assert self.args[0] == self.root.mixed
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['foo', 'bar']
        assert self.args[1].varargs == ['spam', 'eggs']