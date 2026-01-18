import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
class ResourceTest(test_utils.BaseTestCase):

    def test_get_action_args(self):
        env = {'wsgiorg.routing_args': [None, {'controller': None, 'format': None, 'action': 'update', 'id': 12}]}
        expected = {'action': 'update', 'id': 12}
        actual = wsgi.Resource(None, None, None).get_action_args(env)
        self.assertEqual(expected, actual)

    def test_get_action_args_invalid_index(self):
        env = {'wsgiorg.routing_args': []}
        expected = {}
        actual = wsgi.Resource(None, None, None).get_action_args(env)
        self.assertEqual(expected, actual)

    def test_get_action_args_del_controller_error(self):
        actions = {'format': None, 'action': 'update', 'id': 12}
        env = {'wsgiorg.routing_args': [None, actions]}
        expected = {'action': 'update', 'id': 12}
        actual = wsgi.Resource(None, None, None).get_action_args(env)
        self.assertEqual(expected, actual)

    def test_get_action_args_del_format_error(self):
        actions = {'action': 'update', 'id': 12}
        env = {'wsgiorg.routing_args': [None, actions]}
        expected = {'action': 'update', 'id': 12}
        actual = wsgi.Resource(None, None, None).get_action_args(env)
        self.assertEqual(expected, actual)

    def test_dispatch(self):

        class Controller(object):

            def index(self, shirt, pants=None):
                return (shirt, pants)
        resource = wsgi.Resource(None, None, None)
        actual = resource.dispatch(Controller(), 'index', 'on', pants='off')
        expected = ('on', 'off')
        self.assertEqual(expected, actual)

    def test_dispatch_default(self):

        class Controller(object):

            def default(self, shirt, pants=None):
                return (shirt, pants)
        resource = wsgi.Resource(None, None, None)
        actual = resource.dispatch(Controller(), 'index', 'on', pants='off')
        expected = ('on', 'off')
        self.assertEqual(expected, actual)

    def test_dispatch_no_default(self):

        class Controller(object):

            def show(self, shirt, pants=None):
                return (shirt, pants)
        resource = wsgi.Resource(None, None, None)
        self.assertRaises(AttributeError, resource.dispatch, Controller(), 'index', 'on', pants='off')

    def test_dispatch_raises_bad_request(self):

        class FakeController(object):

            def index(self, shirt, pants=None):
                return (shirt, pants)
        resource = wsgi.Resource(FakeController(), None, None)

        def dispatch(self, obj, action, *args, **kwargs):
            raise exception.InvalidPropertyProtectionConfiguration()
        self.mock_object(wsgi.Resource, 'dispatch', dispatch)
        request = wsgi.Request.blank('/')
        self.assertRaises(webob.exc.HTTPBadRequest, resource.__call__, request)

    def test_call(self):

        class FakeController(object):

            def index(self, shirt, pants=None):
                return (shirt, pants)
        resource = wsgi.Resource(FakeController(), None, None)

        def dispatch(self, obj, action, *args, **kwargs):
            if isinstance(obj, wsgi.JSONRequestDeserializer):
                return []
            if isinstance(obj, wsgi.JSONResponseSerializer):
                raise webob.exc.HTTPForbidden()
        self.mock_object(wsgi.Resource, 'dispatch', dispatch)
        request = wsgi.Request.blank('/')
        response = resource.__call__(request)
        self.assertIsInstance(response, webob.exc.HTTPForbidden)
        self.assertEqual(http.FORBIDDEN, response.status_code)

    def test_call_raises_exception(self):

        class FakeController(object):

            def index(self, shirt, pants=None):
                return (shirt, pants)
        resource = wsgi.Resource(FakeController(), None, None)

        def dispatch(self, obj, action, *args, **kwargs):
            raise Exception('test exception')
        self.mock_object(wsgi.Resource, 'dispatch', dispatch)
        request = wsgi.Request.blank('/')
        response = resource.__call__(request)
        self.assertIsInstance(response, webob.exc.HTTPInternalServerError)
        self.assertEqual(http.INTERNAL_SERVER_ERROR, response.status_code)

    @mock.patch.object(wsgi, 'translate_exception')
    def test_resource_call_error_handle_localized(self, mock_translate_exception):

        class Controller(object):

            def delete(self, req, identity):
                raise webob.exc.HTTPBadRequest(explanation='Not Found')
        actions = {'action': 'delete', 'identity': 12}
        env = {'wsgiorg.routing_args': [None, actions]}
        request = wsgi.Request.blank('/tests/123', environ=env)
        message_es = 'No Encontrado'
        resource = wsgi.Resource(Controller(), wsgi.JSONRequestDeserializer(), None)
        translated_exc = webob.exc.HTTPBadRequest(message_es)
        mock_translate_exception.return_value = translated_exc
        e = self.assertRaises(webob.exc.HTTPBadRequest, resource, request)
        self.assertEqual(message_es, str(e))

    @mock.patch.object(webob.acceptparse.AcceptLanguageValidHeader, 'lookup')
    @mock.patch.object(i18n, 'translate')
    def test_translate_exception(self, mock_translate, mock_lookup):
        mock_translate.return_value = 'No Encontrado'
        mock_lookup.return_value = 'de'
        req = wsgi.Request.blank('/tests/123')
        req.headers['Accept-Language'] = 'de'
        e = webob.exc.HTTPNotFound(explanation='Not Found')
        e = wsgi.translate_exception(req, e)
        self.assertEqual('No Encontrado', e.explanation)

    def test_response_headers_encoded(self):
        for_openstack_comrades = 'За опенстек, товарищи'

        class FakeController(object):

            def index(self, shirt, pants=None):
                return (shirt, pants)

        class FakeSerializer(object):

            def index(self, response, result):
                response.headers['unicode_test'] = for_openstack_comrades
        resource = wsgi.Resource(FakeController(), None, FakeSerializer())
        actions = {'action': 'index'}
        env = {'wsgiorg.routing_args': [None, actions]}
        request = wsgi.Request.blank('/tests/123', environ=env)
        response = resource.__call__(request)
        value = response.headers['unicode_test']
        self.assertEqual(for_openstack_comrades, value)