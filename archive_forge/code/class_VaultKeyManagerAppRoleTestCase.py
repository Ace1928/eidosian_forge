import os
import uuid
from oslo_config import cfg
from oslo_utils import uuidutils
from oslotest import base
import requests
from testtools import testcase
from castellan.common import exception
from castellan.key_manager import vault_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
class VaultKeyManagerAppRoleTestCase(VaultKeyManagerTestCase):
    mountpoint = 'secret'

    def _create_key_manager(self):
        key_mgr = vault_key_manager.VaultKeyManager(cfg.CONF)
        if 'VAULT_TEST_URL' not in os.environ or 'VAULT_TEST_ROOT_TOKEN' not in os.environ:
            raise testcase.TestSkipped('Missing Vault setup information')
        self.root_token_id = os.environ['VAULT_TEST_ROOT_TOKEN']
        self.vault_url = os.environ['VAULT_TEST_URL']
        test_uuid = str(uuid.uuid4())
        vault_policy = 'policy-{}'.format(test_uuid)
        vault_approle = 'approle-{}'.format(test_uuid)
        self.session = requests.Session()
        self.session.headers.update({'X-Vault-Token': self.root_token_id})
        self._mount_kv(self.mountpoint)
        self._enable_approle()
        self._create_policy(vault_policy)
        self._create_approle(vault_approle, vault_policy)
        key_mgr._approle_role_id, key_mgr._approle_secret_id = self._retrieve_approle(vault_approle)
        key_mgr._kv_mountpoint = self.mountpoint
        key_mgr._vault_url = self.vault_url
        return key_mgr

    def _mount_kv(self, vault_mountpoint):
        backends = self.session.get('{}/v1/sys/mounts'.format(self.vault_url)).json()
        if vault_mountpoint not in backends:
            params = {'type': 'kv', 'options': {'version': 2}}
            self.session.post('{}/v1/sys/mounts/{}'.format(self.vault_url, vault_mountpoint), json=params)

    def _enable_approle(self):
        params = {'type': 'approle'}
        self.session.post('{}/{}'.format(self.vault_url, AUTH_ENDPOINT.format(auth_type='approle')), json=params)

    def _create_policy(self, vault_policy):
        params = {'rules': TEST_POLICY.format(backend=self.mountpoint)}
        self.session.put('{}/{}'.format(self.vault_url, POLICY_ENDPOINT.format(policy_name=vault_policy)), json=params)

    def _create_approle(self, vault_approle, vault_policy):
        params = {'token_ttl': '60s', 'token_max_ttl': '60s', 'policies': [vault_policy], 'bind_secret_id': 'true', 'bound_cidr_list': '127.0.0.1/32'}
        self.session.post('{}/{}'.format(self.vault_url, APPROLE_ENDPOINT.format(role_name=vault_approle)), json=params)

    def _retrieve_approle(self, vault_approle):
        approle_role_id = self.session.get('{}/v1/auth/approle/role/{}/role-id'.format(self.vault_url, vault_approle)).json()['data']['role_id']
        approle_secret_id = self.session.post('{}/v1/auth/approle/role/{}/secret-id'.format(self.vault_url, vault_approle)).json()['data']['secret_id']
        return (approle_role_id, approle_secret_id)