import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class SessionAuthTests(utils.TestCase):
    TEST_URL = 'http://127.0.0.1:5000/'
    TEST_JSON = {'hello': 'world'}

    def stub_service_url(self, service_type, interface, path, method='GET', **kwargs):
        base_url = AuthPlugin.SERVICE_URLS[service_type][interface]
        uri = '%s/%s' % (base_url.rstrip('/'), path.lstrip('/'))
        self.requests_mock.register_uri(method, uri, **kwargs)

    def test_auth_plugin_default_with_plugin(self):
        self.stub_url('GET', base_url=self.TEST_URL, json=self.TEST_JSON)
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth)
        resp = sess.get(self.TEST_URL)
        self.assertEqual(resp.json(), self.TEST_JSON)
        self.assertRequestHeaderEqual('X-Auth-Token', AuthPlugin.TEST_TOKEN)

    def test_auth_plugin_disable(self):
        self.stub_url('GET', base_url=self.TEST_URL, json=self.TEST_JSON)
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth)
        resp = sess.get(self.TEST_URL, authenticated=False)
        self.assertEqual(resp.json(), self.TEST_JSON)
        self.assertRequestHeaderEqual('X-Auth-Token', None)

    def test_object_delete(self):
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth)
        mock_close = mock.Mock()
        sess._session.close = mock_close
        del sess
        self.assertEqual(1, mock_close.call_count)

    def test_service_type_urls(self):
        service_type = 'compute'
        interface = 'public'
        path = '/instances'
        status = 200
        body = 'SUCCESS'
        self.stub_service_url(service_type=service_type, interface=interface, path=path, status_code=status, text=body)
        sess = client_session.Session(auth=AuthPlugin())
        resp = sess.get(path, endpoint_filter={'service_type': service_type, 'interface': interface})
        self.assertEqual(self.requests_mock.last_request.url, AuthPlugin.SERVICE_URLS['compute']['public'] + path)
        self.assertEqual(resp.text, body)
        self.assertEqual(resp.status_code, status)

    def test_service_url_raises_if_no_auth_plugin(self):
        sess = client_session.Session()
        self.assertRaises(exceptions.MissingAuthPlugin, sess.get, '/path', endpoint_filter={'service_type': 'compute', 'interface': 'public'})

    def test_service_url_raises_if_no_url_returned(self):
        sess = client_session.Session(auth=AuthPlugin())
        self.assertRaises(exceptions.EndpointNotFound, sess.get, '/path', endpoint_filter={'service_type': 'unknown', 'interface': 'public'})

    def test_raises_exc_only_when_asked(self):
        self.requests_mock.get(self.TEST_URL, status_code=401)
        sess = client_session.Session()
        self.assertRaises(exceptions.Unauthorized, sess.get, self.TEST_URL)
        resp = sess.get(self.TEST_URL, raise_exc=False)
        self.assertEqual(401, resp.status_code)

    def test_passed_auth_plugin(self):
        passed = CalledAuthPlugin()
        sess = client_session.Session()
        self.requests_mock.get(CalledAuthPlugin.ENDPOINT + 'path', status_code=200)
        endpoint_filter = {'service_type': 'identity'}
        self.assertRaises(exceptions.MissingAuthPlugin, sess.get, 'path', authenticated=True)
        self.assertRaises(exceptions.MissingAuthPlugin, sess.get, 'path', authenticated=False, endpoint_filter=endpoint_filter)
        resp = sess.get('path', auth=passed, endpoint_filter=endpoint_filter)
        self.assertEqual(200, resp.status_code)
        self.assertTrue(passed.get_endpoint_called)
        self.assertTrue(passed.get_token_called)

    def test_passed_auth_plugin_overrides(self):
        fixed = CalledAuthPlugin()
        passed = CalledAuthPlugin()
        sess = client_session.Session(fixed)
        self.requests_mock.get(CalledAuthPlugin.ENDPOINT + 'path', status_code=200)
        resp = sess.get('path', auth=passed, endpoint_filter={'service_type': 'identity'})
        self.assertEqual(200, resp.status_code)
        self.assertTrue(passed.get_endpoint_called)
        self.assertTrue(passed.get_token_called)
        self.assertFalse(fixed.get_endpoint_called)
        self.assertFalse(fixed.get_token_called)

    def test_requests_auth_plugin(self):
        sess = client_session.Session()
        requests_auth = RequestsAuth()
        self.requests_mock.get(self.TEST_URL, text='resp')
        sess.get(self.TEST_URL, requests_auth=requests_auth)
        last = self.requests_mock.last_request
        self.assertEqual(requests_auth.header_val, last.headers[requests_auth.header_name])
        self.assertTrue(requests_auth.called)

    def test_reauth_called(self):
        auth = CalledAuthPlugin(invalidate=True)
        sess = client_session.Session(auth=auth)
        self.requests_mock.get(self.TEST_URL, [{'text': 'Failed', 'status_code': 401}, {'text': 'Hello', 'status_code': 200}])
        resp = sess.get(self.TEST_URL, authenticated=True)
        self.assertEqual(200, resp.status_code)
        self.assertEqual('Hello', resp.text)
        self.assertTrue(auth.invalidate_called)

    def test_reauth_not_called(self):
        auth = CalledAuthPlugin(invalidate=True)
        sess = client_session.Session(auth=auth)
        self.requests_mock.get(self.TEST_URL, [{'text': 'Failed', 'status_code': 401}, {'text': 'Hello', 'status_code': 200}])
        self.assertRaises(exceptions.Unauthorized, sess.get, self.TEST_URL, authenticated=True, allow_reauth=False)
        self.assertFalse(auth.invalidate_called)

    def test_endpoint_override_overrides_filter(self):
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        override_base = 'http://mytest/'
        path = 'path'
        override_url = override_base + path
        resp_text = uuid.uuid4().hex
        self.requests_mock.get(override_url, text=resp_text)
        resp = sess.get(path, endpoint_override=override_base, endpoint_filter={'service_type': 'identity'})
        self.assertEqual(resp_text, resp.text)
        self.assertEqual(override_url, self.requests_mock.last_request.url)
        self.assertTrue(auth.get_token_called)
        self.assertFalse(auth.get_endpoint_called)
        self.assertFalse(auth.get_user_id_called)
        self.assertFalse(auth.get_project_id_called)

    def test_endpoint_override_ignore_full_url(self):
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        path = 'path'
        url = self.TEST_URL + path
        resp_text = uuid.uuid4().hex
        self.requests_mock.get(url, text=resp_text)
        resp = sess.get(url, endpoint_override='http://someother.url', endpoint_filter={'service_type': 'identity'})
        self.assertEqual(resp_text, resp.text)
        self.assertEqual(url, self.requests_mock.last_request.url)
        self.assertTrue(auth.get_token_called)
        self.assertFalse(auth.get_endpoint_called)
        self.assertFalse(auth.get_user_id_called)
        self.assertFalse(auth.get_project_id_called)

    def test_endpoint_override_does_id_replacement(self):
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        override_base = 'http://mytest/%(project_id)s/%(user_id)s'
        path = 'path'
        replacements = {'user_id': CalledAuthPlugin.USER_ID, 'project_id': CalledAuthPlugin.PROJECT_ID}
        override_url = override_base % replacements + '/' + path
        resp_text = uuid.uuid4().hex
        self.requests_mock.get(override_url, text=resp_text)
        resp = sess.get(path, endpoint_override=override_base, endpoint_filter={'service_type': 'identity'})
        self.assertEqual(resp_text, resp.text)
        self.assertEqual(override_url, self.requests_mock.last_request.url)
        self.assertTrue(auth.get_token_called)
        self.assertTrue(auth.get_user_id_called)
        self.assertTrue(auth.get_project_id_called)
        self.assertFalse(auth.get_endpoint_called)

    def test_endpoint_override_fails_to_replace_if_none(self):
        auth = token_endpoint.Token(uuid.uuid4().hex, uuid.uuid4().hex)
        sess = client_session.Session(auth=auth)
        override_base = 'http://mytest/%(project_id)s'
        e = self.assertRaises(ValueError, sess.get, '/path', endpoint_override=override_base, endpoint_filter={'service_type': 'identity'})
        self.assertIn('project_id', str(e))
        override_base = 'http://mytest/%(user_id)s'
        e = self.assertRaises(ValueError, sess.get, '/path', endpoint_override=override_base, endpoint_filter={'service_type': 'identity'})
        self.assertIn('user_id', str(e))

    def test_endpoint_override_fails_to_do_unknown_replacement(self):
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        override_base = 'http://mytest/%(unknown_id)s'
        e = self.assertRaises(AttributeError, sess.get, '/path', endpoint_override=override_base, endpoint_filter={'service_type': 'identity'})
        self.assertIn('unknown_id', str(e))

    def test_user_and_project_id(self):
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth)
        self.assertEqual(auth.TEST_USER_ID, sess.get_user_id())
        self.assertEqual(auth.TEST_PROJECT_ID, sess.get_project_id())

    def test_logger_object_passed(self):
        logger = logging.getLogger(uuid.uuid4().hex)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        string_io = io.StringIO()
        handler = logging.StreamHandler(string_io)
        logger.addHandler(handler)
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth)
        response = {uuid.uuid4().hex: uuid.uuid4().hex}
        self.stub_url('GET', json=response, headers={'Content-Type': 'application/json'})
        resp = sess.get(self.TEST_URL, logger=logger)
        self.assertEqual(response, resp.json())
        output = string_io.getvalue()
        self.assertIn(self.TEST_URL, output)
        self.assertIn(list(response.keys())[0], output)
        self.assertIn(list(response.values())[0], output)
        self.assertNotIn(list(response.keys())[0], self.logger.output)
        self.assertNotIn(list(response.values())[0], self.logger.output)

    def test_split_loggers(self):

        def get_logger_io(name):
            logger_name = 'keystoneauth.session.{name}'.format(name=name)
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            string_io = io.StringIO()
            handler = logging.StreamHandler(string_io)
            logger.addHandler(handler)
            return string_io
        io_dict = {}
        for name in ('request', 'body', 'response', 'request-id'):
            io_dict[name] = get_logger_io(name)
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth, split_loggers=True)
        response_key = uuid.uuid4().hex
        response_val = uuid.uuid4().hex
        response = {response_key: response_val}
        request_id = uuid.uuid4().hex
        self.stub_url('GET', json=response, headers={'Content-Type': 'application/json', 'X-OpenStack-Request-ID': request_id})
        resp = sess.get(self.TEST_URL, headers={encodeutils.safe_encode('x-bytes-header'): encodeutils.safe_encode('bytes-value')})
        self.assertEqual(response, resp.json())
        request_output = io_dict['request'].getvalue().strip()
        response_output = io_dict['response'].getvalue().strip()
        body_output = io_dict['body'].getvalue().strip()
        id_output = io_dict['request-id'].getvalue().strip()
        self.assertIn('curl -g -i -X GET {url}'.format(url=self.TEST_URL), request_output)
        self.assertIn('-H "x-bytes-header: bytes-value"', request_output)
        self.assertEqual('[200] Content-Type: application/json X-OpenStack-Request-ID: {id}'.format(id=request_id), response_output)
        self.assertEqual('GET call to {url} used request id {id}'.format(url=self.TEST_URL, id=request_id), id_output)
        self.assertEqual('{{"{key}": "{val}"}}'.format(key=response_key, val=response_val), body_output)

    def test_collect_timing(self):
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth, collect_timing=True)
        response = {uuid.uuid4().hex: uuid.uuid4().hex}
        self.stub_url('GET', json=response, headers={'Content-Type': 'application/json'})
        resp = sess.get(self.TEST_URL)
        self.assertEqual(response, resp.json())
        timings = sess.get_timings()
        self.assertEqual(timings[0].method, 'GET')
        self.assertEqual(timings[0].url, self.TEST_URL)
        self.assertIsInstance(timings[0].elapsed, datetime.timedelta)
        sess.reset_timings()
        timings = sess.get_timings()
        self.assertEqual(len(timings), 0)