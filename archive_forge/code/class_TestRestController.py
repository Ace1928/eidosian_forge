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
class TestRestController(PecanTestCase):

    @property
    def app_(self):

        class OthersController(object):

            @expose()
            def index(self, req, resp):
                return 'OTHERS'

            @expose()
            def echo(self, req, resp, value):
                return str(value)

        class ThingsController(RestController):
            data = ['zero', 'one', 'two', 'three']
            _custom_actions = {'count': ['GET'], 'length': ['GET', 'POST']}
            others = OthersController()

            @expose()
            def get_one(self, req, resp, id):
                return self.data[int(id)]

            @expose('json')
            def get_all(self, req, resp):
                return dict(items=self.data)

            @expose()
            def length(self, req, resp, id, value=None):
                length = len(self.data[int(id)])
                if value:
                    length += len(value)
                return str(length)

            @expose()
            def post(self, req, resp, value):
                self.data.append(value)
                resp.status = 302
                return 'CREATED'

            @expose()
            def edit(self, req, resp, id):
                return 'EDIT %s' % self.data[int(id)]

            @expose()
            def put(self, req, resp, id, value):
                self.data[int(id)] = value
                return 'UPDATED'

            @expose()
            def get_delete(self, req, resp, id):
                return 'DELETE %s' % self.data[int(id)]

            @expose()
            def delete(self, req, resp, id):
                del self.data[int(id)]
                return 'DELETED'

            @expose()
            def trace(self, req, resp):
                return 'TRACE'

            @expose()
            def post_options(self, req, resp):
                return 'OPTIONS'

            @expose()
            def options(self, req, resp):
                abort(500)

            @expose()
            def other(self, req, resp):
                abort(500)

        class RootController(object):
            things = ThingsController()
        return TestApp(Pecan(RootController(), use_context_locals=False))

    def test_get_all(self):
        r = self.app_.get('/things')
        assert r.status_int == 200
        assert r.body == dumps(dict(items=['zero', 'one', 'two', 'three'])).encode('utf-8')

    def test_get_one(self):
        for i, value in enumerate([b'zero', b'one', b'two', b'three']):
            r = self.app_.get('/things/%d' % i)
            assert r.status_int == 200
            assert r.body == value

    def test_post(self):
        r = self.app_.post('/things', {'value': 'four'})
        assert r.status_int == 302
        assert r.body == b'CREATED'

    def test_custom_action(self):
        r = self.app_.get('/things/3/edit')
        assert r.status_int == 200
        assert r.body == b'EDIT three'

    def test_put(self):
        r = self.app_.put('/things/3', {'value': 'THREE!'})
        assert r.status_int == 200
        assert r.body == b'UPDATED'

    def test_put_with_method_parameter_and_get(self):
        r = self.app_.get('/things/3?_method=put', {'value': 'X'}, status=405)
        assert r.status_int == 405

    def test_put_with_method_parameter_and_post(self):
        r = self.app_.post('/things/3?_method=put', {'value': 'THREE!'})
        assert r.status_int == 200
        assert r.body == b'UPDATED'

    def test_get_delete(self):
        r = self.app_.get('/things/3/delete')
        assert r.status_int == 200
        assert r.body == b'DELETE three'

    def test_delete_method(self):
        r = self.app_.delete('/things/3')
        assert r.status_int == 200
        assert r.body == b'DELETED'

    def test_delete_with_method_parameter(self):
        r = self.app_.get('/things/3?_method=DELETE', status=405)
        assert r.status_int == 405

    def test_delete_with_method_parameter_and_post(self):
        r = self.app_.post('/things/3?_method=DELETE')
        assert r.status_int == 200
        assert r.body == b'DELETED'

    def test_custom_method_type(self):
        r = self.app_.request('/things', method='TRACE')
        assert r.status_int == 200
        assert r.body == b'TRACE'

    def test_custom_method_type_with_method_parameter(self):
        r = self.app_.get('/things?_method=TRACE')
        assert r.status_int == 200
        assert r.body == b'TRACE'

    def test_options(self):
        r = self.app_.request('/things', method='OPTIONS')
        assert r.status_int == 200
        assert r.body == b'OPTIONS'

    def test_options_with_method_parameter(self):
        r = self.app_.post('/things', {'_method': 'OPTIONS'})
        assert r.status_int == 200
        assert r.body == b'OPTIONS'

    def test_other_custom_action(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = self.app_.request('/things/other', method='MISC', status=405)
            assert r.status_int == 405

    def test_other_custom_action_with_method_parameter(self):
        r = self.app_.post('/things/other', {'_method': 'MISC'}, status=405)
        assert r.status_int == 405

    def test_nested_controller_with_trailing_slash(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = self.app_.request('/things/others/', method='MISC')
            assert r.status_int == 200
            assert r.body == b'OTHERS'

    def test_nested_controller_without_trailing_slash(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = self.app_.request('/things/others', method='MISC', status=302)
            assert r.status_int == 302

    def test_invalid_custom_action(self):
        r = self.app_.get('/things?_method=BAD', status=405)
        assert r.status_int == 405

    def test_named_action(self):
        r = self.app_.get('/things/1/length')
        assert r.status_int == 200
        assert r.body == b'3'

    def test_named_nested_action(self):
        r = self.app_.get('/things/others/echo?value=test')
        assert r.status_int == 200
        assert r.body == b'test'

    def test_nested_post(self):
        r = self.app_.post('/things/others/echo', {'value': 'test'})
        assert r.status_int == 200
        assert r.body == b'test'