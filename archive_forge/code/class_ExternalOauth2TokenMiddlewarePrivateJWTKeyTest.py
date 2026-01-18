import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
class ExternalOauth2TokenMiddlewarePrivateJWTKeyTest(BaseExternalOauth2TokenMiddlewareTest):

    def setUp(self):
        super(ExternalOauth2TokenMiddlewarePrivateJWTKeyTest, self).setUp()
        self._test_client_id = str(uuid.uuid4())
        self._test_client_secret = str(uuid.uuid4())
        self._jwt_key_file = '/root/key.pem'
        self._auth_method = 'private_key_jwt'
        self._test_conf = get_config(introspect_endpoint=self._introspect_endpoint, audience=self._audience, auth_method=self._auth_method, client_id=self._test_client_id, client_secret=self._test_client_secret, jwt_key_file=self._jwt_key_file, jwt_algorithm='RS256', jwt_bearer_time_out='2800', mapping_project_id='access_project.id', mapping_project_name='access_project.name', mapping_project_domain_id='access_project.domain.id', mapping_project_domain_name='access_project.domain.name', mapping_user_id='client_id', mapping_user_name='username', mapping_user_domain_id='user_domain.id', mapping_user_domain_name='user_domain.name', mapping_roles='roles')
        self._token = str(uuid.uuid4()) + '_user_token'
        self._user_id = str(uuid.uuid4()) + '_user_id'
        self._user_name = str(uuid.uuid4()) + '_user_name'
        self._user_domain_id = str(uuid.uuid4()) + '_user_domain_id'
        self._user_domain_name = str(uuid.uuid4()) + '_user_domain_name'
        self._project_id = str(uuid.uuid4()) + '_project_id'
        self._project_name = str(uuid.uuid4()) + '_project_name'
        self._project_domain_id = str(uuid.uuid4()) + 'project_domain_id'
        self._project_domain_name = str(uuid.uuid4()) + 'project_domain_name'
        self._roles = 'admin,member,reader'
        self._default_metadata = {'access_project': {'id': self._project_id, 'name': self._project_name, 'domain': {'id': self._project_domain_id, 'name': self._project_domain_name}}, 'user_domain': {'id': self._user_domain_id, 'name': self._user_domain_name}, 'roles': self._roles, 'client_id': self._user_id, 'username': self._user_name}

    @mock.patch('os.path.isfile')
    @mock.patch('builtins.open', mock.mock_open(read_data=JWT_KEY_CONTENT))
    def test_basic_200(self, mocker_path_isfile):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)

        def mocker_isfile_side_effect(filename):
            if filename == self._jwt_key_file:
                return True
            else:
                return False
        mocker_path_isfile.side_effect = mocker_isfile_side_effect
        resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
        self.assertTrue(mocker_path_isfile.called)
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self._check_env_value_project_scope(resp.request.environ, self._user_id, self._user_name, self._user_domain_id, self._user_domain_name, self._project_id, self._project_name, self._project_domain_id, self._project_domain_name, self._roles)

    @mock.patch('os.path.isfile')
    @mock.patch('builtins.open', mock.mock_open(read_data=JWT_KEY_CONTENT))
    def test_introspect_by_private_key_jwt_error_alg_500(self, mocker_path_isfile):
        conf = copy.deepcopy(self._test_conf)
        conf['jwt_algorithm'] = 'HS256'
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)

        def mocker_isfile_side_effect(filename):
            if filename == self._jwt_key_file:
                return True
            else:
                return False
        mocker_path_isfile.side_effect = mocker_isfile_side_effect
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    @mock.patch('os.path.isfile')
    @mock.patch('builtins.open', mock.mock_open(read_data=''))
    def test_introspect_by_private_key_jwt_error_file_no_content_500(self, mocker_path_isfile):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)

        def mocker_isfile_side_effect(filename):
            if filename == self._jwt_key_file:
                return True
            else:
                return False
        mocker_path_isfile.side_effect = mocker_isfile_side_effect
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    @mock.patch('os.path.isfile')
    def test_introspect_by_private_key_jwt_error_file_can_not_read_500(self, mocker_path_isfile):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)

        def mocker_isfile_side_effect(filename):
            if filename == self._jwt_key_file:
                return True
            else:
                return False
        mocker_path_isfile.side_effect = mocker_isfile_side_effect
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})

    def test_introspect_by_private_key_jwt_error_file_not_exist_500(self):
        conf = copy.deepcopy(self._test_conf)
        self.set_middleware(conf=conf)

        def mock_resp(request, context):
            return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
        self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
        self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
        self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})