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
class DockerFileStore(object):
    """
    A custom credential store class that implements only the functionality we need to
    update the docker config file when no credential helpers is provided.
    """
    program = '<legacy config>'

    def __init__(self, config_path):
        self._config_path = config_path
        self._config = dict(auths=dict())
        try:
            with open(self._config_path, 'r') as f:
                config = json.load(f)
        except (ValueError, IOError):
            config = dict()
        self._config.update(config)

    @property
    def config_path(self):
        """
        Return the config path configured in this DockerFileStore instance.
        """
        return self._config_path

    def get(self, server):
        """
        Retrieve credentials for `server` if there are any in the config file.
        Otherwise raise a `StoreError`
        """
        server_creds = self._config['auths'].get(server)
        if not server_creds:
            raise CredentialsNotFound('No matching credentials')
        username, password = decode_auth(server_creds['auth'])
        return dict(Username=username, Secret=password)

    def _write(self):
        """
        Write config back out to disk.
        """
        dir = os.path.dirname(self._config_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        content = json.dumps(self._config, indent=4, sort_keys=True).encode('utf-8')
        f = os.open(self._config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 384)
        try:
            os.write(f, content)
        finally:
            os.close(f)

    def store(self, server, username, password):
        """
        Add a credentials for `server` to the current configuration.
        """
        b64auth = base64.b64encode(to_bytes(username) + b':' + to_bytes(password))
        auth = to_text(b64auth)
        if 'auths' not in self._config:
            self._config['auths'] = dict()
        self._config['auths'][server] = dict(auth=auth)
        self._write()

    def erase(self, server):
        """
        Remove credentials for the given server from the configuration.
        """
        if 'auths' in self._config and server in self._config['auths']:
            self._config['auths'].pop(server)
            self._write()