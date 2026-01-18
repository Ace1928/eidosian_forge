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
class AdapterTest(utils.TestCase):
    SERVICE_TYPE = uuid.uuid4().hex
    SERVICE_NAME = uuid.uuid4().hex
    INTERFACE = uuid.uuid4().hex
    REGION_NAME = uuid.uuid4().hex
    USER_AGENT = uuid.uuid4().hex
    VERSION = uuid.uuid4().hex
    ALLOW = {'allow_deprecated': False, 'allow_experimental': True, 'allow_unknown': True}
    TEST_URL = CalledAuthPlugin.ENDPOINT

    def _create_loaded_adapter(self, sess=None, auth=None):
        return adapter.Adapter(sess or client_session.Session(), auth=auth or CalledAuthPlugin(), service_type=self.SERVICE_TYPE, service_name=self.SERVICE_NAME, interface=self.INTERFACE, region_name=self.REGION_NAME, user_agent=self.USER_AGENT, version=self.VERSION, allow=self.ALLOW)

    def _verify_endpoint_called(self, adpt):
        self.assertEqual(self.SERVICE_TYPE, adpt.auth.endpoint_arguments['service_type'])
        self.assertEqual(self.SERVICE_NAME, adpt.auth.endpoint_arguments['service_name'])
        self.assertEqual(self.INTERFACE, adpt.auth.endpoint_arguments['interface'])
        self.assertEqual(self.REGION_NAME, adpt.auth.endpoint_arguments['region_name'])
        self.assertEqual(self.VERSION, adpt.auth.endpoint_arguments['version'])

    def test_setting_variables_on_request(self):
        response = uuid.uuid4().hex
        self.stub_url('GET', text=response)
        adpt = self._create_loaded_adapter()
        resp = adpt.get('/')
        self.assertEqual(resp.text, response)
        self._verify_endpoint_called(adpt)
        self.assertEqual(self.ALLOW, adpt.auth.endpoint_arguments['allow'])
        self.assertTrue(adpt.auth.get_token_called)
        self.assertRequestHeaderEqual('User-Agent', self.USER_AGENT)

    def test_setting_global_id_on_request(self):
        global_id_adpt = 'req-%s' % uuid.uuid4()
        global_id_req = 'req-%s' % uuid.uuid4()
        response = uuid.uuid4().hex
        self.stub_url('GET', text=response)

        def mk_adpt(**kwargs):
            return adapter.Adapter(client_session.Session(), auth=CalledAuthPlugin(), service_type=self.SERVICE_TYPE, service_name=self.SERVICE_NAME, interface=self.INTERFACE, region_name=self.REGION_NAME, user_agent=self.USER_AGENT, version=self.VERSION, allow=self.ALLOW, **kwargs)
        adpt = mk_adpt()
        resp = adpt.get('/')
        self.assertEqual(resp.text, response)
        self._verify_endpoint_called(adpt)
        self.assertEqual(self.ALLOW, adpt.auth.endpoint_arguments['allow'])
        self.assertTrue(adpt.auth.get_token_called)
        self.assertRequestHeaderEqual('X-OpenStack-Request-ID', None)
        adpt.get('/', global_request_id=global_id_req)
        self.assertRequestHeaderEqual('X-OpenStack-Request-ID', global_id_req)
        adpt = mk_adpt(global_request_id=global_id_adpt)
        adpt.get('/')
        self.assertRequestHeaderEqual('X-OpenStack-Request-ID', global_id_adpt)
        adpt.get('/', global_request_id=global_id_req)
        self.assertRequestHeaderEqual('X-OpenStack-Request-ID', global_id_req)

    def test_setting_variables_on_get_endpoint(self):
        adpt = self._create_loaded_adapter()
        url = adpt.get_endpoint()
        self.assertEqual(self.TEST_URL, url)
        self._verify_endpoint_called(adpt)

    def test_legacy_binding(self):
        key = uuid.uuid4().hex
        val = uuid.uuid4().hex
        response = json.dumps({key: val})
        self.stub_url('GET', text=response)
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        adpt = adapter.LegacyJsonAdapter(sess, service_type=self.SERVICE_TYPE, user_agent=self.USER_AGENT)
        resp, body = adpt.get('/')
        self.assertEqual(self.SERVICE_TYPE, auth.endpoint_arguments['service_type'])
        self.assertEqual(resp.text, response)
        self.assertEqual(val, body[key])

    def test_legacy_binding_non_json_resp(self):
        response = uuid.uuid4().hex
        self.stub_url('GET', text=response, headers={'Content-Type': 'text/html'})
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        adpt = adapter.LegacyJsonAdapter(sess, service_type=self.SERVICE_TYPE, user_agent=self.USER_AGENT)
        resp, body = adpt.get('/')
        self.assertEqual(self.SERVICE_TYPE, auth.endpoint_arguments['service_type'])
        self.assertEqual(resp.text, response)
        self.assertIsNone(body)

    def test_methods(self):
        sess = client_session.Session()
        adpt = adapter.Adapter(sess)
        url = 'http://url'
        for method in ['get', 'head', 'post', 'put', 'patch', 'delete']:
            with mock.patch.object(adpt, 'request') as m:
                getattr(adpt, method)(url)
                m.assert_called_once_with(url, method.upper())

    def test_setting_endpoint_override(self):
        endpoint_override = 'http://overrideurl'
        path = '/path'
        endpoint_url = endpoint_override + path
        auth = CalledAuthPlugin()
        sess = client_session.Session(auth=auth)
        adpt = adapter.Adapter(sess, endpoint_override=endpoint_override)
        response = uuid.uuid4().hex
        self.requests_mock.get(endpoint_url, text=response)
        resp = adpt.get(path)
        self.assertEqual(response, resp.text)
        self.assertEqual(endpoint_url, self.requests_mock.last_request.url)
        self.assertEqual(endpoint_override, adpt.get_endpoint())

    def test_adapter_invalidate(self):
        auth = CalledAuthPlugin()
        sess = client_session.Session()
        adpt = adapter.Adapter(sess, auth=auth)
        adpt.invalidate()
        self.assertTrue(auth.invalidate_called)

    def test_adapter_get_token(self):
        auth = CalledAuthPlugin()
        sess = client_session.Session()
        adpt = adapter.Adapter(sess, auth=auth)
        self.assertEqual(self.TEST_TOKEN, adpt.get_token())
        self.assertTrue(auth.get_token_called)

    def test_adapter_connect_retries(self):
        retries = 2
        sess = client_session.Session()
        adpt = adapter.Adapter(sess, connect_retries=retries)
        self.stub_url('GET', exc=requests.exceptions.ConnectionError())
        with mock.patch('time.sleep') as m:
            self.assertRaises(exceptions.ConnectionError, adpt.get, self.TEST_URL)
            self.assertEqual(retries, m.call_count)
        self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))

    def test_adapter_http_503_retries(self):
        retries = 2
        sess = client_session.Session()
        adpt = adapter.Adapter(sess, status_code_retries=retries)
        self.stub_url('GET', status_code=503)
        with mock.patch('time.sleep') as m:
            self.assertRaises(exceptions.ServiceUnavailable, adpt.get, self.TEST_URL)
            self.assertEqual(retries, m.call_count)
        self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))

    def test_adapter_http_status_retries(self):
        retries = 2
        sess = client_session.Session()
        adpt = adapter.Adapter(sess, status_code_retries=retries, retriable_status_codes=[503, 409])
        self.stub_url('GET', status_code=409)
        with mock.patch('time.sleep') as m:
            self.assertRaises(exceptions.Conflict, adpt.get, self.TEST_URL)
            self.assertEqual(retries, m.call_count)
        self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))

    def test_user_and_project_id(self):
        auth = AuthPlugin()
        sess = client_session.Session()
        adpt = adapter.Adapter(sess, auth=auth)
        self.assertEqual(auth.TEST_USER_ID, adpt.get_user_id())
        self.assertEqual(auth.TEST_PROJECT_ID, adpt.get_project_id())

    def test_logger_object_passed(self):
        logger = logging.getLogger(uuid.uuid4().hex)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        string_io = io.StringIO()
        handler = logging.StreamHandler(string_io)
        logger.addHandler(handler)
        auth = AuthPlugin()
        sess = client_session.Session(auth=auth)
        adpt = adapter.Adapter(sess, auth=auth, logger=logger)
        response = {uuid.uuid4().hex: uuid.uuid4().hex}
        self.stub_url('GET', json=response, headers={'Content-Type': 'application/json'})
        resp = adpt.get(self.TEST_URL, logger=logger)
        self.assertEqual(response, resp.json())
        output = string_io.getvalue()
        self.assertIn(self.TEST_URL, output)
        self.assertIn(list(response.keys())[0], output)
        self.assertIn(list(response.values())[0], output)
        self.assertNotIn(list(response.keys())[0], self.logger.output)
        self.assertNotIn(list(response.values())[0], self.logger.output)

    def test_unknown_connection_error(self):
        self.stub_url('GET', exc=requests.exceptions.RequestException)
        self.assertRaises(exceptions.UnknownConnectionError, client_session.Session().request, self.TEST_URL, 'GET')

    def test_additional_headers(self):
        session_key = uuid.uuid4().hex
        session_val = uuid.uuid4().hex
        adapter_key = uuid.uuid4().hex
        adapter_val = uuid.uuid4().hex
        request_key = uuid.uuid4().hex
        request_val = uuid.uuid4().hex
        text = uuid.uuid4().hex
        url = 'http://keystone.test.com'
        self.requests_mock.get(url, text=text)
        sess = client_session.Session(additional_headers={session_key: session_val})
        adap = adapter.Adapter(session=sess, additional_headers={adapter_key: adapter_val})
        resp = adap.get(url, headers={request_key: request_val})
        request = self.requests_mock.last_request
        self.assertEqual(resp.text, text)
        self.assertEqual(session_val, request.headers[session_key])
        self.assertEqual(adapter_val, request.headers[adapter_key])
        self.assertEqual(request_val, request.headers[request_key])

    def test_additional_headers_overrides(self):
        header = uuid.uuid4().hex
        session_val = uuid.uuid4().hex
        adapter_val = uuid.uuid4().hex
        request_val = uuid.uuid4().hex
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        sess = client_session.Session(additional_headers={header: session_val})
        adap = adapter.Adapter(session=sess)
        adap.get(url)
        self.assertEqual(session_val, self.requests_mock.last_request.headers[header])
        adap.additional_headers[header] = adapter_val
        adap.get(url)
        self.assertEqual(adapter_val, self.requests_mock.last_request.headers[header])
        adap.get(url, headers={header: request_val})
        self.assertEqual(request_val, self.requests_mock.last_request.headers[header])

    def test_adapter_user_agent_session_adapter(self):
        sess = client_session.Session(app_name='ksatest', app_version='1.2.3')
        adap = adapter.Adapter(client_name='testclient', client_version='4.5.6', session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'ksatest/1.2.3 testclient/4.5.6'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_session_version_on_adapter(self):

        class TestAdapter(adapter.Adapter):
            client_name = 'testclient'
            client_version = '4.5.6'
        sess = client_session.Session(app_name='ksatest', app_version='1.2.3')
        adap = TestAdapter(session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'ksatest/1.2.3 testclient/4.5.6'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_session_adapter_no_app_version(self):
        sess = client_session.Session(app_name='ksatest')
        adap = adapter.Adapter(client_name='testclient', client_version='4.5.6', session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'ksatest testclient/4.5.6'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_session_adapter_no_client_version(self):
        sess = client_session.Session(app_name='ksatest', app_version='1.2.3')
        adap = adapter.Adapter(client_name='testclient', session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'ksatest/1.2.3 testclient'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_session_adapter_additional(self):
        sess = client_session.Session(app_name='ksatest', app_version='1.2.3', additional_user_agent=[('one', '1.1.1'), ('two', '2.2.2')])
        adap = adapter.Adapter(client_name='testclient', client_version='4.5.6', session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'ksatest/1.2.3 testclient/4.5.6 one/1.1.1 two/2.2.2'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_session(self):
        sess = client_session.Session(app_name='ksatest', app_version='1.2.3')
        adap = adapter.Adapter(session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'ksatest/1.2.3'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_adapter(self):
        sess = client_session.Session()
        adap = adapter.Adapter(client_name='testclient', client_version='4.5.6', session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        adap.get(url)
        agent = 'testclient/4.5.6'
        self.assertEqual(agent + ' ' + client_session.DEFAULT_USER_AGENT, self.requests_mock.last_request.headers['User-Agent'])

    def test_adapter_user_agent_session_override(self):
        sess = client_session.Session(app_name='ksatest', app_version='1.2.3', additional_user_agent=[('one', '1.1.1'), ('two', '2.2.2')])
        adap = adapter.Adapter(client_name='testclient', client_version='4.5.6', session=sess)
        url = 'http://keystone.test.com'
        self.requests_mock.get(url)
        override_user_agent = '%s/%s' % (uuid.uuid4().hex, uuid.uuid4().hex)
        adap.get(url, user_agent=override_user_agent)
        self.assertEqual(override_user_agent, self.requests_mock.last_request.headers['User-Agent'])

    def test_nested_adapters(self):
        text = uuid.uuid4().hex
        token = uuid.uuid4().hex
        url = 'http://keystone.example.com/path'
        sess = client_session.Session()
        auth = CalledAuthPlugin()
        auth.ENDPOINT = url
        auth.TOKEN = token
        adap1 = adapter.Adapter(session=sess, interface='public')
        adap2 = adapter.Adapter(session=adap1, service_type='identity', auth=auth)
        self.requests_mock.get(url + '/test', text=text)
        resp = adap2.get('/test')
        self.assertEqual(text, resp.text)
        self.assertTrue(auth.get_endpoint_called)
        self.assertEqual('public', auth.endpoint_arguments['interface'])
        self.assertEqual('identity', auth.endpoint_arguments['service_type'])
        last_token = self.requests_mock.last_request.headers['X-Auth-Token']
        self.assertEqual(token, last_token)

    def test_default_microversion(self):
        sess = client_session.Session()
        url = 'http://url'

        def validate(adap_kwargs, get_kwargs, exp_kwargs):
            with mock.patch.object(sess, 'request') as m:
                adapter.Adapter(sess, **adap_kwargs).get(url, **get_kwargs)
                m.assert_called_once_with(url, 'GET', endpoint_filter={}, headers={}, rate_semaphore=mock.ANY, **exp_kwargs)
        validate({}, {}, {})
        validate({'default_microversion': '1.2'}, {}, {'microversion': '1.2'})
        validate({}, {'microversion': '1.2'}, {'microversion': '1.2'})
        validate({'default_microversion': '1.2'}, {'microversion': '1.5'}, {'microversion': '1.5'})

    def test_raise_exc_override(self):
        sess = client_session.Session()
        url = 'http://url'

        def validate(adap_kwargs, get_kwargs, exp_kwargs):
            with mock.patch.object(sess, 'request') as m:
                adapter.Adapter(sess, **adap_kwargs).get(url, **get_kwargs)
                m.assert_called_once_with(url, 'GET', endpoint_filter={}, headers={}, rate_semaphore=mock.ANY, **exp_kwargs)
        validate({}, {}, {})
        validate({'raise_exc': True}, {}, {'raise_exc': True})
        validate({'raise_exc': False}, {}, {'raise_exc': False})
        validate({}, {'raise_exc': True}, {'raise_exc': True})
        validate({}, {'raise_exc': False}, {'raise_exc': False})
        validate({'raise_exc': True}, {'raise_exc': False}, {'raise_exc': False})
        validate({'raise_exc': False}, {'raise_exc': True}, {'raise_exc': True})