import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
class TestRestControllerStateAccess(PecanTestCase):

    def setUp(self):
        super(TestRestControllerStateAccess, self).setUp()
        self.args = None

        class RootController(rest.RestController):

            @expose()
            def _default(self, _id, *args, **kw):
                return 'Default'

            @expose()
            def get_all(self, **kw):
                return 'All'

            @expose()
            def get_one(self, _id, *args, **kw):
                return 'One'

            @expose()
            def post(self, *args, **kw):
                return 'POST'

            @expose()
            def put(self, _id, *args, **kw):
                return 'PUT'

            @expose()
            def delete(self, _id, *args, **kw):
                return 'DELETE'

        class SimpleHook(PecanHook):

            def before(inself, state):
                self.args = (state.controller, state.arguments)
        self.root = RootController()
        self.app = TestApp(make_app(self.root, hooks=[SimpleHook()]))

    def test_get_all(self):
        self.app.get('/')
        assert self.args[0] == self.root.get_all
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_get_all_with_kwargs(self):
        self.app.get('/?foo=bar')
        assert self.args[0] == self.root.get_all
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'foo': 'bar'}

    def test_get_one(self):
        self.app.get('/1')
        assert self.args[0] == self.root.get_one
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_get_one_with_varargs(self):
        self.app.get('/1/2/3')
        assert self.args[0] == self.root.get_one
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == ['2', '3']
        assert kwargs(self.args[1]) == {}

    def test_get_one_with_kwargs(self):
        self.app.get('/1?foo=bar')
        assert self.args[0] == self.root.get_one
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'foo': 'bar'}

    def test_post(self):
        self.app.post('/')
        assert self.args[0] == self.root.post
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_post_with_varargs(self):
        self.app.post('/foo/bar')
        assert self.args[0] == self.root.post
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == ['foo', 'bar']
        assert kwargs(self.args[1]) == {}

    def test_post_with_kwargs(self):
        self.app.post('/', params={'foo': 'bar'})
        assert self.args[0] == self.root.post
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == []
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'foo': 'bar'}

    def test_put(self):
        self.app.put('/1')
        assert self.args[0] == self.root.put
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_put_with_method_argument(self):
        self.app.post('/1?_method=put')
        assert self.args[0] == self.root.put
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'_method': 'put'}

    def test_put_with_varargs(self):
        self.app.put('/1/2/3')
        assert self.args[0] == self.root.put
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == ['2', '3']
        assert kwargs(self.args[1]) == {}

    def test_put_with_kwargs(self):
        self.app.put('/1?foo=bar')
        assert self.args[0] == self.root.put
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'foo': 'bar'}

    def test_delete(self):
        self.app.delete('/1')
        assert self.args[0] == self.root.delete
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {}

    def test_delete_with_method_argument(self):
        self.app.post('/1?_method=delete')
        assert self.args[0] == self.root.delete
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'_method': 'delete'}

    def test_delete_with_varargs(self):
        self.app.delete('/1/2/3')
        assert self.args[0] == self.root.delete
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == ['2', '3']
        assert kwargs(self.args[1]) == {}

    def test_delete_with_kwargs(self):
        self.app.delete('/1?foo=bar')
        assert self.args[0] == self.root.delete
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'foo': 'bar'}

    def test_post_with_invalid_method_kwarg(self):
        self.app.post('/1?_method=invalid')
        assert self.args[0] == self.root._default
        assert isinstance(self.args[1], inspect.Arguments)
        assert self.args[1].args == ['1']
        assert self.args[1].varargs == []
        assert kwargs(self.args[1]) == {'_method': 'invalid'}