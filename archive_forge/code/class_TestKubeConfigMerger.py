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
class TestKubeConfigMerger(BaseTestCase):
    TEST_KUBE_CONFIG_PART1 = {'current-context': 'no_user', 'contexts': [{'name': 'no_user', 'context': {'cluster': 'default'}}], 'clusters': [{'name': 'default', 'cluster': {'server': TEST_HOST}}], 'users': []}
    TEST_KUBE_CONFIG_PART2 = {'current-context': '', 'contexts': [{'name': 'ssl', 'context': {'cluster': 'ssl', 'user': 'ssl'}}, {'name': 'simple_token', 'context': {'cluster': 'default', 'user': 'simple_token'}}], 'clusters': [{'name': 'ssl', 'cluster': {'server': TEST_SSL_HOST, 'certificate-authority-data': TEST_CERTIFICATE_AUTH_BASE64}}], 'users': [{'name': 'ssl', 'user': {'token': TEST_DATA_BASE64, 'client-certificate-data': TEST_CLIENT_CERT_BASE64, 'client-key-data': TEST_CLIENT_KEY_BASE64}}]}
    TEST_KUBE_CONFIG_PART3 = {'current-context': 'no_user', 'contexts': [{'name': 'expired_oidc', 'context': {'cluster': 'default', 'user': 'expired_oidc'}}, {'name': 'ssl', 'context': {'cluster': 'skipped-part2-defined-this-context', 'user': 'skipped'}}], 'clusters': [], 'users': [{'name': 'expired_oidc', 'user': {'auth-provider': {'name': 'oidc', 'config': {'client-id': 'tectonic-kubectl', 'client-secret': 'FAKE_SECRET', 'id-token': TEST_OIDC_EXPIRED_LOGIN, 'idp-certificate-authority-data': TEST_OIDC_CA, 'idp-issuer-url': 'https://example.org/identity', 'refresh-token': 'lucWJjEhlxZW01cXI3YmVlcYnpxNGhzk'}}}}, {'name': 'simple_token', 'user': {'token': TEST_DATA_BASE64, 'username': TEST_USERNAME, 'password': TEST_PASSWORD}}]}

    def _create_multi_config(self):
        files = []
        for part in (self.TEST_KUBE_CONFIG_PART1, self.TEST_KUBE_CONFIG_PART2, self.TEST_KUBE_CONFIG_PART3):
            files.append(self._create_temp_file(yaml.safe_dump(part)))
        return ENV_KUBECONFIG_PATH_SEPARATOR.join(files)

    def test_list_kube_config_contexts(self):
        kubeconfigs = self._create_multi_config()
        expected_contexts = [{'context': {'cluster': 'default'}, 'name': 'no_user'}, {'context': {'cluster': 'ssl', 'user': 'ssl'}, 'name': 'ssl'}, {'context': {'cluster': 'default', 'user': 'simple_token'}, 'name': 'simple_token'}, {'context': {'cluster': 'default', 'user': 'expired_oidc'}, 'name': 'expired_oidc'}]
        contexts, active_context = list_kube_config_contexts(config_file=kubeconfigs)
        self.assertEqual(contexts, expected_contexts)
        self.assertEqual(active_context, expected_contexts[0])

    def test_new_client_from_config(self):
        kubeconfigs = self._create_multi_config()
        client = new_client_from_config(config_file=kubeconfigs, context='simple_token')
        self.assertEqual(TEST_HOST, client.configuration.host)
        self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, client.configuration.api_key['authorization'])

    def test_save_changes(self):
        kubeconfigs = self._create_multi_config()
        kconf = KubeConfigMerger(kubeconfigs)
        user = kconf.config['users'].get_with_name('expired_oidc')['user']
        provider = user['auth-provider']['config']
        provider.value['id-token'] = 'token-changed'
        kconf.save_changes()
        kconf = KubeConfigMerger(kubeconfigs)
        user = kconf.config['users'].get_with_name('expired_oidc')['user']
        provider = user['auth-provider']['config']
        self.assertEqual(provider.value['id-token'], 'token-changed')