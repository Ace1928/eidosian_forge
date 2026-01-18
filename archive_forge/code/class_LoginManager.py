from __future__ import absolute_import, division, print_function
import base64
import json
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api import auth
from ansible_collections.community.docker.plugins.module_utils._api.auth import decode_auth
from ansible_collections.community.docker.plugins.module_utils._api.credentials.errors import CredentialsNotFound
from ansible_collections.community.docker.plugins.module_utils._api.credentials.store import Store
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
class LoginManager(DockerBaseClass):

    def __init__(self, client, results):
        super(LoginManager, self).__init__()
        self.client = client
        self.results = results
        parameters = self.client.module.params
        self.check_mode = self.client.check_mode
        self.registry_url = parameters.get('registry_url')
        self.username = parameters.get('username')
        self.password = parameters.get('password')
        self.reauthorize = parameters.get('reauthorize')
        self.config_path = parameters.get('config_path')
        self.state = parameters.get('state')

    def run(self):
        """
        Do the actual work of this task here. This allows instantiation for partial
        testing.
        """
        if self.state == 'present':
            self.login()
        else:
            self.logout()

    def fail(self, msg):
        self.client.fail(msg)

    def _login(self, reauth):
        if self.config_path and os.path.exists(self.config_path):
            self.client._auth_configs = auth.load_config(self.config_path, credstore_env=self.client.credstore_env)
        elif not self.client._auth_configs or self.client._auth_configs.is_empty:
            self.client._auth_configs = auth.load_config(credstore_env=self.client.credstore_env)
        authcfg = self.client._auth_configs.resolve_authconfig(self.registry_url)
        if authcfg and authcfg.get('username', None) == self.username and (not reauth):
            return authcfg
        req_data = {'username': self.username, 'password': self.password, 'email': None, 'serveraddress': self.registry_url}
        response = self.client._post_json(self.client._url('/auth'), data=req_data)
        if response.status_code == 200:
            self.client._auth_configs.add_auth(self.registry_url or auth.INDEX_NAME, req_data)
        return self.client._result(response, json=True)

    def login(self):
        """
        Log into the registry with provided username/password. On success update the config
        file with the new authorization.

        :return: None
        """
        self.results['actions'].append('Logged into %s' % self.registry_url)
        self.log('Log into %s with username %s' % (self.registry_url, self.username))
        try:
            response = self._login(self.reauthorize)
        except Exception as exc:
            self.fail('Logging into %s for user %s failed - %s' % (self.registry_url, self.username, to_native(exc)))
        if 'password' in response:
            if not self.reauthorize and response['password'] != self.password:
                try:
                    response = self._login(True)
                except Exception as exc:
                    self.fail('Logging into %s for user %s failed - %s' % (self.registry_url, self.username, to_native(exc)))
            response.pop('password', None)
        self.results['login_result'] = response
        self.update_credentials()

    def logout(self):
        """
        Log out of the registry. On success update the config file.

        :return: None
        """
        store = self.get_credential_store_instance(self.registry_url, self.config_path)
        try:
            store.get(self.registry_url)
        except CredentialsNotFound:
            self.log('Credentials for %s not present, doing nothing.' % self.registry_url)
            self.results['changed'] = False
            return
        if not self.check_mode:
            store.erase(self.registry_url)
        self.results['changed'] = True

    def update_credentials(self):
        """
        If the authorization is not stored attempt to store authorization values via
        the appropriate credential helper or to the config file.

        :return: None
        """
        store = self.get_credential_store_instance(self.registry_url, self.config_path)
        try:
            current = store.get(self.registry_url)
        except CredentialsNotFound:
            current = dict(Username='', Secret='')
        if current['Username'] != self.username or current['Secret'] != self.password or self.reauthorize:
            if not self.check_mode:
                store.store(self.registry_url, self.username, self.password)
            self.log('Writing credentials to configured helper %s for %s' % (store.program, self.registry_url))
            self.results['actions'].append('Wrote credentials to configured helper %s for %s' % (store.program, self.registry_url))
            self.results['changed'] = True

    def get_credential_store_instance(self, registry, dockercfg_path):
        """
        Return an instance of docker.credentials.Store used by the given registry.

        :return: A Store or None
        :rtype: Union[docker.credentials.Store, NoneType]
        """
        credstore_env = self.client.credstore_env
        config = auth.load_config(config_path=dockercfg_path)
        store_name = auth.get_credential_store(config, registry)
        if store_name:
            self.log('Found credential store %s' % store_name)
            return Store(store_name, environment=credstore_env)
        return DockerFileStore(dockercfg_path)