import base64
import datetime
import json
import os
import shutil
import tempfile
import unittest
import mock
from ruamel import yaml
from six import PY3, next
from kubernetes.client import Configuration
from .config_exception import ConfigException
from .kube_config import (ENV_KUBECONFIG_PATH_SEPARATOR, ConfigNode, FileOrData,
class TestKubeConfigLoader(BaseTestCase):
    TEST_KUBE_CONFIG = {'current-context': 'no_user', 'contexts': [{'name': 'no_user', 'context': {'cluster': 'default'}}, {'name': 'simple_token', 'context': {'cluster': 'default', 'user': 'simple_token'}}, {'name': 'gcp', 'context': {'cluster': 'default', 'user': 'gcp'}}, {'name': 'expired_gcp', 'context': {'cluster': 'default', 'user': 'expired_gcp'}}, {'name': 'expired_gcp_refresh', 'context': {'cluster': 'default', 'user': 'expired_gcp_refresh'}}, {'name': 'oidc', 'context': {'cluster': 'default', 'user': 'oidc'}}, {'name': 'expired_oidc', 'context': {'cluster': 'default', 'user': 'expired_oidc'}}, {'name': 'expired_oidc_nocert', 'context': {'cluster': 'default', 'user': 'expired_oidc_nocert'}}, {'name': 'oidc_contains_reserved_character', 'context': {'cluster': 'default', 'user': 'oidc_contains_reserved_character'}}, {'name': 'oidc_invalid_padding_length', 'context': {'cluster': 'default', 'user': 'oidc_invalid_padding_length'}}, {'name': 'user_pass', 'context': {'cluster': 'default', 'user': 'user_pass'}}, {'name': 'ssl', 'context': {'cluster': 'ssl', 'user': 'ssl'}}, {'name': 'no_ssl_verification', 'context': {'cluster': 'no_ssl_verification', 'user': 'ssl'}}, {'name': 'ssl-no_file', 'context': {'cluster': 'ssl-no_file', 'user': 'ssl-no_file'}}, {'name': 'ssl-local-file', 'context': {'cluster': 'ssl-local-file', 'user': 'ssl-local-file'}}, {'name': 'non_existing_user', 'context': {'cluster': 'default', 'user': 'non_existing_user'}}, {'name': 'exec_cred_user', 'context': {'cluster': 'default', 'user': 'exec_cred_user'}}], 'clusters': [{'name': 'default', 'cluster': {'server': TEST_HOST}}, {'name': 'ssl-no_file', 'cluster': {'server': TEST_SSL_HOST, 'certificate-authority': TEST_CERTIFICATE_AUTH}}, {'name': 'ssl-local-file', 'cluster': {'server': TEST_SSL_HOST, 'certificate-authority': 'cert_test'}}, {'name': 'ssl', 'cluster': {'server': TEST_SSL_HOST, 'certificate-authority-data': TEST_CERTIFICATE_AUTH_BASE64}}, {'name': 'no_ssl_verification', 'cluster': {'server': TEST_SSL_HOST, 'insecure-skip-tls-verify': 'true'}}], 'users': [{'name': 'simple_token', 'user': {'token': TEST_DATA_BASE64, 'username': TEST_USERNAME, 'password': TEST_PASSWORD}}, {'name': 'gcp', 'user': {'auth-provider': {'name': 'gcp', 'config': {'access-token': TEST_DATA_BASE64}}, 'token': TEST_DATA_BASE64, 'username': TEST_USERNAME, 'password': TEST_PASSWORD}}, {'name': 'expired_gcp', 'user': {'auth-provider': {'name': 'gcp', 'config': {'access-token': TEST_DATA_BASE64, 'expiry': TEST_TOKEN_EXPIRY_PAST}}, 'token': TEST_DATA_BASE64, 'username': TEST_USERNAME, 'password': TEST_PASSWORD}}, {'name': 'expired_gcp_refresh', 'user': {'auth-provider': {'name': 'gcp', 'config': {'access-token': TEST_DATA_BASE64, 'expiry': TEST_TOKEN_EXPIRY_PAST}}, 'token': TEST_DATA_BASE64, 'username': TEST_USERNAME, 'password': TEST_PASSWORD}}, {'name': 'oidc', 'user': {'auth-provider': {'name': 'oidc', 'config': {'id-token': TEST_OIDC_LOGIN}}}}, {'name': 'expired_oidc', 'user': {'auth-provider': {'name': 'oidc', 'config': {'client-id': 'tectonic-kubectl', 'client-secret': 'FAKE_SECRET', 'id-token': TEST_OIDC_EXPIRED_LOGIN, 'idp-certificate-authority-data': TEST_OIDC_CA, 'idp-issuer-url': 'https://example.org/identity', 'refresh-token': 'lucWJjEhlxZW01cXI3YmVlcYnpxNGhzk'}}}}, {'name': 'expired_oidc_nocert', 'user': {'auth-provider': {'name': 'oidc', 'config': {'client-id': 'tectonic-kubectl', 'client-secret': 'FAKE_SECRET', 'id-token': TEST_OIDC_EXPIRED_LOGIN, 'idp-issuer-url': 'https://example.org/identity', 'refresh-token': 'lucWJjEhlxZW01cXI3YmVlcYnpxNGhzk'}}}}, {'name': 'oidc_contains_reserved_character', 'user': {'auth-provider': {'name': 'oidc', 'config': {'client-id': 'tectonic-kubectl', 'client-secret': 'FAKE_SECRET', 'id-token': TEST_OIDC_CONTAINS_RESERVED_CHARACTERS, 'idp-issuer-url': 'https://example.org/identity', 'refresh-token': 'lucWJjEhlxZW01cXI3YmVlcYnpxNGhzk'}}}}, {'name': 'oidc_invalid_padding_length', 'user': {'auth-provider': {'name': 'oidc', 'config': {'client-id': 'tectonic-kubectl', 'client-secret': 'FAKE_SECRET', 'id-token': TEST_OIDC_INVALID_PADDING_LENGTH, 'idp-issuer-url': 'https://example.org/identity', 'refresh-token': 'lucWJjEhlxZW01cXI3YmVlcYnpxNGhzk'}}}}, {'name': 'user_pass', 'user': {'username': TEST_USERNAME, 'password': TEST_PASSWORD}}, {'name': 'ssl-no_file', 'user': {'token': TEST_DATA_BASE64, 'client-certificate': TEST_CLIENT_CERT, 'client-key': TEST_CLIENT_KEY}}, {'name': 'ssl-local-file', 'user': {'tokenFile': 'token_file', 'client-certificate': 'client_cert', 'client-key': 'client_key'}}, {'name': 'ssl', 'user': {'token': TEST_DATA_BASE64, 'client-certificate-data': TEST_CLIENT_CERT_BASE64, 'client-key-data': TEST_CLIENT_KEY_BASE64}}, {'name': 'exec_cred_user', 'user': {'exec': {'apiVersion': 'client.authentication.k8s.io/v1beta1', 'command': 'aws-iam-authenticator', 'args': ['token', '-i', 'dummy-cluster']}}}]}

    def test_no_user_context(self):
        expected = FakeConfig(host=TEST_HOST)
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='no_user').load_and_set(actual)
        self.assertEqual(expected, actual)

    def test_simple_token(self):
        expected = FakeConfig(host=TEST_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64)
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='simple_token').load_and_set(actual)
        self.assertEqual(expected, actual)

    def test_load_user_token(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='simple_token')
        self.assertTrue(loader._load_user_token())
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, loader.token)

    def test_gcp_no_refresh(self):
        fake_config = FakeConfig()
        self.assertFalse(hasattr(fake_config, 'get_api_key_with_prefix'))
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='gcp', get_google_credentials=lambda: _raise_exception('SHOULD NOT BE CALLED')).load_and_set(fake_config)
        self.assertIsNotNone(fake_config.get_api_key_with_prefix)
        self.assertEqual(TEST_HOST, fake_config.host)
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, fake_config.api_key['authorization'])

    def test_load_gcp_token_no_refresh(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='gcp', get_google_credentials=lambda: _raise_exception('SHOULD NOT BE CALLED'))
        self.assertTrue(loader._load_auth_provider_token())
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, loader.token)

    def test_load_gcp_token_with_refresh(self):

        def cred():
            return None
        cred.token = TEST_ANOTHER_DATA_BASE64
        cred.expiry = datetime.datetime.utcnow()
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_gcp', get_google_credentials=lambda: cred)
        original_expiry = _get_expiry(loader, 'expired_gcp')
        self.assertTrue(loader._load_auth_provider_token())
        new_expiry = _get_expiry(loader, 'expired_gcp')
        self.assertTrue(new_expiry > original_expiry)
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_ANOTHER_DATA_BASE64, loader.token)

    def test_gcp_get_api_key_with_prefix(self):

        class cred_old:
            token = TEST_DATA_BASE64
            expiry = DATETIME_EXPIRY_PAST

        class cred_new:
            token = TEST_ANOTHER_DATA_BASE64
            expiry = DATETIME_EXPIRY_FUTURE
        fake_config = FakeConfig()
        _get_google_credentials = mock.Mock()
        _get_google_credentials.side_effect = [cred_old, cred_new]
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_gcp_refresh', get_google_credentials=_get_google_credentials)
        loader.load_and_set(fake_config)
        original_expiry = _get_expiry(loader, 'expired_gcp_refresh')
        token = fake_config.get_api_key_with_prefix()
        new_expiry = _get_expiry(loader, 'expired_gcp_refresh')
        self.assertTrue(new_expiry > original_expiry)
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_ANOTHER_DATA_BASE64, loader.token)
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_ANOTHER_DATA_BASE64, token)

    def test_oidc_no_refresh(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='oidc')
        self.assertTrue(loader._load_auth_provider_token())
        self.assertEqual(TEST_OIDC_TOKEN, loader.token)

    @mock.patch('kubernetes.config.kube_config.OAuth2Session.refresh_token')
    @mock.patch('kubernetes.config.kube_config.ApiClient.request')
    def test_oidc_with_refresh(self, mock_ApiClient, mock_OAuth2Session):
        mock_response = mock.MagicMock()
        type(mock_response).status = mock.PropertyMock(return_value=200)
        type(mock_response).data = mock.PropertyMock(return_value=json.dumps({'token_endpoint': 'https://example.org/identity/token'}))
        mock_ApiClient.return_value = mock_response
        mock_OAuth2Session.return_value = {'id_token': 'abc123', 'refresh_token': 'newtoken123'}
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_oidc')
        self.assertTrue(loader._load_auth_provider_token())
        self.assertEqual('Bearer abc123', loader.token)

    @mock.patch('kubernetes.config.kube_config.OAuth2Session.refresh_token')
    @mock.patch('kubernetes.config.kube_config.ApiClient.request')
    def test_oidc_with_refresh_nocert(self, mock_ApiClient, mock_OAuth2Session):
        mock_response = mock.MagicMock()
        type(mock_response).status = mock.PropertyMock(return_value=200)
        type(mock_response).data = mock.PropertyMock(return_value=json.dumps({'token_endpoint': 'https://example.org/identity/token'}))
        mock_ApiClient.return_value = mock_response
        mock_OAuth2Session.return_value = {'id_token': 'abc123', 'refresh_token': 'newtoken123'}
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_oidc_nocert')
        self.assertTrue(loader._load_auth_provider_token())
        self.assertEqual('Bearer abc123', loader.token)

    def test_oidc_fails_if_contains_reserved_chars(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='oidc_contains_reserved_character')
        self.assertEqual(loader._load_oid_token('oidc_contains_reserved_character'), None)

    def test_oidc_fails_if_invalid_padding_length(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='oidc_invalid_padding_length')
        self.assertEqual(loader._load_oid_token('oidc_invalid_padding_length'), None)

    def test_user_pass(self):
        expected = FakeConfig(host=TEST_HOST, token=TEST_BASIC_TOKEN)
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='user_pass').load_and_set(actual)
        self.assertEqual(expected, actual)

    def test_load_user_pass_token(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='user_pass')
        self.assertTrue(loader._load_user_pass_token())
        self.assertEqual(TEST_BASIC_TOKEN, loader.token)

    def test_ssl_no_cert_files(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='ssl-no_file')
        self.expect_exception(loader.load_and_set, 'does not exists', FakeConfig())

    def test_ssl(self):
        expected = FakeConfig(host=TEST_SSL_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, cert_file=self._create_temp_file(TEST_CLIENT_CERT), key_file=self._create_temp_file(TEST_CLIENT_KEY), ssl_ca_cert=self._create_temp_file(TEST_CERTIFICATE_AUTH))
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='ssl').load_and_set(actual)
        self.assertEqual(expected, actual)

    def test_ssl_no_verification(self):
        expected = FakeConfig(host=TEST_SSL_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, cert_file=self._create_temp_file(TEST_CLIENT_CERT), key_file=self._create_temp_file(TEST_CLIENT_KEY), verify_ssl=False, ssl_ca_cert=None)
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='no_ssl_verification').load_and_set(actual)
        self.assertEqual(expected, actual)

    def test_list_contexts(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='no_user')
        actual_contexts = loader.list_contexts()
        expected_contexts = ConfigNode('', self.TEST_KUBE_CONFIG)['contexts']
        for actual in actual_contexts:
            expected = expected_contexts.get_with_name(actual['name'])
            self.assertEqual(expected.value, actual)

    def test_current_context(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG)
        expected_contexts = ConfigNode('', self.TEST_KUBE_CONFIG)['contexts']
        self.assertEqual(expected_contexts.get_with_name('no_user').value, loader.current_context)

    def test_set_active_context(self):
        loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG)
        loader.set_active_context('ssl')
        expected_contexts = ConfigNode('', self.TEST_KUBE_CONFIG)['contexts']
        self.assertEqual(expected_contexts.get_with_name('ssl').value, loader.current_context)

    def test_ssl_with_relative_ssl_files(self):
        expected = FakeConfig(host=TEST_SSL_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, cert_file=self._create_temp_file(TEST_CLIENT_CERT), key_file=self._create_temp_file(TEST_CLIENT_KEY), ssl_ca_cert=self._create_temp_file(TEST_CERTIFICATE_AUTH))
        try:
            temp_dir = tempfile.mkdtemp()
            actual = FakeConfig()
            with open(os.path.join(temp_dir, 'cert_test'), 'wb') as fd:
                fd.write(TEST_CERTIFICATE_AUTH.encode())
            with open(os.path.join(temp_dir, 'client_cert'), 'wb') as fd:
                fd.write(TEST_CLIENT_CERT.encode())
            with open(os.path.join(temp_dir, 'client_key'), 'wb') as fd:
                fd.write(TEST_CLIENT_KEY.encode())
            with open(os.path.join(temp_dir, 'token_file'), 'wb') as fd:
                fd.write(TEST_DATA_BASE64.encode())
            KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='ssl-local-file', config_base_path=temp_dir).load_and_set(actual)
            self.assertEqual(expected, actual)
        finally:
            shutil.rmtree(temp_dir)

    def test_load_kube_config(self):
        expected = FakeConfig(host=TEST_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64)
        config_file = self._create_temp_file(yaml.safe_dump(self.TEST_KUBE_CONFIG))
        actual = FakeConfig()
        load_kube_config(config_file=config_file, context='simple_token', client_configuration=actual)
        self.assertEqual(expected, actual)

    def test_list_kube_config_contexts(self):
        config_file = self._create_temp_file(yaml.safe_dump(self.TEST_KUBE_CONFIG))
        contexts, active_context = list_kube_config_contexts(config_file=config_file)
        self.assertDictEqual(self.TEST_KUBE_CONFIG['contexts'][0], active_context)
        if PY3:
            self.assertCountEqual(self.TEST_KUBE_CONFIG['contexts'], contexts)
        else:
            self.assertItemsEqual(self.TEST_KUBE_CONFIG['contexts'], contexts)

    def test_new_client_from_config(self):
        config_file = self._create_temp_file(yaml.safe_dump(self.TEST_KUBE_CONFIG))
        client = new_client_from_config(config_file=config_file, context='simple_token')
        self.assertEqual(TEST_HOST, client.configuration.host)
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, client.configuration.api_key['authorization'])

    def test_no_users_section(self):
        expected = FakeConfig(host=TEST_HOST)
        actual = FakeConfig()
        test_kube_config = self.TEST_KUBE_CONFIG.copy()
        del test_kube_config['users']
        KubeConfigLoader(config_dict=test_kube_config, active_context='gcp').load_and_set(actual)
        self.assertEqual(expected, actual)

    def test_non_existing_user(self):
        expected = FakeConfig(host=TEST_HOST)
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='non_existing_user').load_and_set(actual)
        self.assertEqual(expected, actual)

    @mock.patch('kubernetes.config.kube_config.ExecProvider.run')
    def test_user_exec_auth(self, mock):
        token = 'dummy'
        mock.return_value = {'token': token}
        expected = FakeConfig(host=TEST_HOST, api_key={'authorization': BEARER_TOKEN_FORMAT % token})
        actual = FakeConfig()
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='exec_cred_user').load_and_set(actual)
        self.assertEqual(expected, actual)