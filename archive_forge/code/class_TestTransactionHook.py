import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
class TestTransactionHook(PecanTestCase):

    def test_transaction_hook(self):
        run_hook = []

        class RootController(object):

            @expose()
            def index(self):
                run_hook.append('inside')
                return 'Hello, World!'

            @expose()
            def redirect(self):
                redirect('/')

            @expose()
            def error(self):
                return [][1]

        def gen(event):
            return lambda: run_hook.append(event)
        app = TestApp(make_app(RootController(), hooks=[TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert len(run_hook) == 3
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'clear'
        run_hook = []
        response = app.post('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert len(run_hook) == 4
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'commit'
        assert run_hook[3] == 'clear'
        run_hook = []
        response = app.get('/redirect')
        assert response.status_int == 302
        assert len(run_hook) == 2
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        run_hook = []
        response = app.post('/redirect')
        assert response.status_int == 302
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'commit'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.post('/error')
        except IndexError:
            pass
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'

    def test_transaction_hook_with_after_actions(self):
        run_hook = []

        def action(name):

            def action_impl():
                run_hook.append(name)
            return action_impl

        class RootController(object):

            @expose()
            @after_commit(action('action-one'))
            def index(self):
                run_hook.append('inside')
                return 'Index Method!'

            @expose()
            @transactional()
            @after_commit(action('action-two'))
            def decorated(self):
                run_hook.append('inside')
                return 'Decorated Method!'

            @expose()
            @after_rollback(action('action-three'))
            def rollback(self):
                abort(500)

            @expose()
            @transactional()
            @after_rollback(action('action-four'))
            def rollback_decorated(self):
                abort(500)

        def gen(event):
            return lambda: run_hook.append(event)
        app = TestApp(make_app(RootController(), hooks=[TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Index Method!'
        assert len(run_hook) == 3
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'clear'
        run_hook = []
        response = app.post('/')
        assert response.status_int == 200
        assert response.body == b'Index Method!'
        assert len(run_hook) == 5
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'commit'
        assert run_hook[3] == 'action-one'
        assert run_hook[4] == 'clear'
        run_hook = []
        response = app.get('/decorated')
        assert response.status_int == 200
        assert response.body == b'Decorated Method!'
        assert len(run_hook) == 7
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'inside'
        assert run_hook[4] == 'commit'
        assert run_hook[5] == 'action-two'
        assert run_hook[6] == 'clear'
        run_hook = []
        response = app.get('/rollback', expect_errors=True)
        assert response.status_int == 500
        assert len(run_hook) == 2
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        run_hook = []
        response = app.post('/rollback', expect_errors=True)
        assert response.status_int == 500
        assert len(run_hook) == 4
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'action-three'
        assert run_hook[3] == 'clear'
        run_hook = []
        response = app.get('/rollback_decorated', expect_errors=True)
        assert response.status_int == 500
        assert len(run_hook) == 6
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'rollback'
        assert run_hook[4] == 'action-four'
        assert run_hook[5] == 'clear'
        run_hook = []
        response = app.get('/fourohfour', status=404)
        assert response.status_int == 404
        assert len(run_hook) == 2
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'

    def test_transaction_hook_with_transactional_decorator(self):
        run_hook = []

        class RootController(object):

            @expose()
            def index(self):
                run_hook.append('inside')
                return 'Hello, World!'

            @expose()
            def redirect(self):
                redirect('/')

            @expose()
            @transactional()
            def redirect_transactional(self):
                redirect('/')

            @expose()
            @transactional(False)
            def redirect_rollback(self):
                redirect('/')

            @expose()
            def error(self):
                return [][1]

            @expose()
            @transactional(False)
            def error_rollback(self):
                return [][1]

            @expose()
            @transactional()
            def error_transactional(self):
                return [][1]

        def gen(event):
            return lambda: run_hook.append(event)
        app = TestApp(make_app(RootController(), hooks=[TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert len(run_hook) == 3
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'clear'
        run_hook = []
        response = app.post('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert len(run_hook) == 4
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'commit'
        assert run_hook[3] == 'clear'
        run_hook = []
        response = app.get('/redirect')
        assert response.status_int == 302
        assert len(run_hook) == 2
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        run_hook = []
        response = app.post('/redirect')
        assert response.status_int == 302
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'commit'
        assert run_hook[2] == 'clear'
        run_hook = []
        response = app.get('/redirect_transactional')
        assert response.status_int == 302
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'commit'
        assert run_hook[4] == 'clear'
        run_hook = []
        response = app.post('/redirect_transactional')
        assert response.status_int == 302
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'commit'
        assert run_hook[2] == 'clear'
        run_hook = []
        response = app.get('/redirect_rollback')
        assert response.status_int == 302
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'rollback'
        assert run_hook[4] == 'clear'
        run_hook = []
        response = app.post('/redirect_rollback')
        assert response.status_int == 302
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.post('/error')
        except IndexError:
            pass
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.get('/error')
        except IndexError:
            pass
        assert len(run_hook) == 2
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        run_hook = []
        try:
            response = app.post('/error_transactional')
        except IndexError:
            pass
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.get('/error_transactional')
        except IndexError:
            pass
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'rollback'
        assert run_hook[4] == 'clear'
        run_hook = []
        try:
            response = app.post('/error_rollback')
        except IndexError:
            pass
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.get('/error_rollback')
        except IndexError:
            pass
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'rollback'
        assert run_hook[4] == 'clear'

    def test_transaction_hook_with_transactional_class_decorator(self):
        run_hook = []

        @transactional()
        class RootController(object):

            @expose()
            def index(self):
                run_hook.append('inside')
                return 'Hello, World!'

            @expose()
            def redirect(self):
                redirect('/')

            @expose()
            @transactional(False)
            def redirect_rollback(self):
                redirect('/')

            @expose()
            def error(self):
                return [][1]

            @expose(generic=True)
            def generic(self):
                pass

            @generic.when(method='GET')
            def generic_get(self):
                run_hook.append('inside')
                return 'generic get'

            @generic.when(method='POST')
            def generic_post(self):
                run_hook.append('inside')
                return 'generic post'

        def gen(event):
            return lambda: run_hook.append(event)
        app = TestApp(make_app(RootController(), hooks=[TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
        response = app.get('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert len(run_hook) == 6
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'inside'
        assert run_hook[4] == 'commit'
        assert run_hook[5] == 'clear'
        run_hook = []
        response = app.post('/')
        assert response.status_int == 200
        assert response.body == b'Hello, World!'
        assert len(run_hook) == 4
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'commit'
        assert run_hook[3] == 'clear'
        run_hook = []
        response = app.get('/redirect')
        assert response.status_int == 302
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'commit'
        assert run_hook[4] == 'clear'
        run_hook = []
        response = app.post('/redirect')
        assert response.status_int == 302
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'commit'
        assert run_hook[2] == 'clear'
        run_hook = []
        response = app.get('/redirect_rollback')
        assert response.status_int == 302
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'rollback'
        assert run_hook[4] == 'clear'
        run_hook = []
        response = app.post('/redirect_rollback')
        assert response.status_int == 302
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.post('/error')
        except IndexError:
            pass
        assert len(run_hook) == 3
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'rollback'
        assert run_hook[2] == 'clear'
        run_hook = []
        try:
            response = app.get('/error')
        except IndexError:
            pass
        assert len(run_hook) == 5
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'rollback'
        assert run_hook[4] == 'clear'
        run_hook = []
        response = app.get('/generic')
        assert response.status_int == 200
        assert response.body == b'generic get'
        assert len(run_hook) == 6
        assert run_hook[0] == 'start_ro'
        assert run_hook[1] == 'clear'
        assert run_hook[2] == 'start'
        assert run_hook[3] == 'inside'
        assert run_hook[4] == 'commit'
        assert run_hook[5] == 'clear'
        run_hook = []
        response = app.post('/generic')
        assert response.status_int == 200
        assert response.body == b'generic post'
        assert len(run_hook) == 4
        assert run_hook[0] == 'start'
        assert run_hook[1] == 'inside'
        assert run_hook[2] == 'commit'
        assert run_hook[3] == 'clear'

    def test_transaction_hook_with_broken_hook(self):
        """
        In a scenario where a preceding hook throws an exception,
        ensure that TransactionHook still rolls back properly.
        """
        run_hook = []

        class RootController(object):

            @expose()
            def index(self):
                return 'Hello, World!'

        def gen(event):
            return lambda: run_hook.append(event)

        class MyCustomException(Exception):
            pass

        class MyHook(PecanHook):

            def on_route(self, state):
                raise MyCustomException('BROKEN!')
        app = TestApp(make_app(RootController(), hooks=[MyHook(), TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
        self.assertRaises(MyCustomException, app.get, '/')
        assert len(run_hook) == 1
        assert run_hook[0] == 'clear'