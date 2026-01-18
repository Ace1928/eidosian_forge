from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
class S3TokenMiddlewareTestGood(S3TokenMiddlewareTestBase):

    def setUp(self):
        super(S3TokenMiddlewareTestGood, self).setUp()
        self.middleware = s3_token.S3Token(FakeApp(), self.conf)
        self.requests_mock.post(self.TEST_URL, status_code=201, json=GOOD_RESPONSE)

    def test_no_path_request(self):
        req = webob.Request.blank('/')
        self.middleware(req.environ, self.start_fake_response)
        self.assertEqual(self.response_status, 200)

    def test_without_authorization(self):
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        self.middleware(req.environ, self.start_fake_response)
        self.assertEqual(self.response_status, 200)

    def test_without_auth_storage_token(self):
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'badboy'
        self.middleware(req.environ, self.start_fake_response)
        self.assertEqual(self.response_status, 200)

    def test_authorized(self):
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'access:signature'
        req.headers['X-Storage-Token'] = 'token'
        req.get_response(self.middleware)
        self.assertTrue(req.path.startswith('/v1/AUTH_TENANT_ID'))
        self.assertEqual(req.headers['X-Auth-Token'], 'TOKEN_ID')

    def test_authorized_http(self):
        protocol = 'http'
        host = 'fakehost'
        port = 35357
        self.requests_mock.post('%s://%s:%s/v2.0/s3tokens' % (protocol, host, port), status_code=201, json=GOOD_RESPONSE)
        self.middleware = s3_token.filter_factory({'auth_protocol': protocol, 'auth_host': host, 'auth_port': port})(FakeApp())
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'access:signature'
        req.headers['X-Storage-Token'] = 'token'
        req.get_response(self.middleware)
        self.assertTrue(req.path.startswith('/v1/AUTH_TENANT_ID'))
        self.assertEqual(req.headers['X-Auth-Token'], 'TOKEN_ID')

    def test_authorization_nova_toconnect(self):
        req = webob.Request.blank('/v1/AUTH_swiftint/c/o')
        req.headers['Authorization'] = 'access:FORCED_TENANT_ID:signature'
        req.headers['X-Storage-Token'] = 'token'
        req.get_response(self.middleware)
        path = req.environ['PATH_INFO']
        self.assertTrue(path.startswith('/v1/AUTH_FORCED_TENANT_ID'))

    @mock.patch.object(requests, 'post')
    def test_insecure(self, MOCK_REQUEST):
        self.middleware = s3_token.filter_factory({'insecure': 'True'})(FakeApp())
        text_return_value = jsonutils.dumps(GOOD_RESPONSE).encode()
        MOCK_REQUEST.return_value = utils.TestResponse({'status_code': 201, 'text': text_return_value})
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'access:signature'
        req.headers['X-Storage-Token'] = 'token'
        req.get_response(self.middleware)
        self.assertTrue(MOCK_REQUEST.called)
        mock_args, mock_kwargs = MOCK_REQUEST.call_args
        self.assertIs(mock_kwargs['verify'], False)

    def test_insecure_option(self):
        true_values = ['true', 'True', '1', 'yes']
        for val in true_values:
            config = {'insecure': val, 'certfile': 'false_ind'}
            middleware = s3_token.filter_factory(config)(FakeApp())
            self.assertIs(False, middleware._verify)
        false_values = ['false', 'False', '0', 'no', 'someweirdvalue']
        for val in false_values:
            config = {'insecure': val, 'certfile': 'false_ind'}
            middleware = s3_token.filter_factory(config)(FakeApp())
            self.assertEqual('false_ind', middleware._verify)
        config = {'certfile': 'false_ind'}
        middleware = s3_token.filter_factory(config)(FakeApp())
        self.assertIs('false_ind', middleware._verify)

    def test_unicode_path(self):
        url = u'/v1/AUTH_cfa/c/euro€'.encode('utf8')
        req = webob.Request.blank(urllib.parse.quote(url))
        req.headers['Authorization'] = 'access:signature'
        req.headers['X-Storage-Token'] = 'token'
        req.get_response(self.middleware)