from __future__ import (absolute_import, division, print_function)
import base64
import os
import json
from stat import S_IRUSR, S_IWUSR
from ansible import constants as C
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
class GalaxyToken(object):
    """ Class to storing and retrieving local galaxy token """
    token_type = 'Token'

    def __init__(self, token=None):
        self.b_file = to_bytes(C.GALAXY_TOKEN_PATH, errors='surrogate_or_strict')
        self._config = None
        self._token = token

    @property
    def config(self):
        if self._config is None:
            self._config = self._read()
        if self._token:
            self._config['token'] = None if self._token is NoTokenSentinel else self._token
        return self._config

    def _read(self):
        action = 'Opened'
        if not os.path.isfile(self.b_file):
            open(self.b_file, 'w').close()
            os.chmod(self.b_file, S_IRUSR | S_IWUSR)
            action = 'Created'
        with open(self.b_file, 'r') as f:
            config = yaml_load(f)
        display.vvv('%s %s' % (action, to_text(self.b_file)))
        if config and (not isinstance(config, dict)):
            display.vvv('Galaxy token file %s malformed, unable to read it' % to_text(self.b_file))
            return {}
        return config or {}

    def set(self, token):
        self._token = token
        self.save()

    def get(self):
        return self.config.get('token', None)

    def save(self):
        with open(self.b_file, 'w') as f:
            yaml_dump(self.config, f, default_flow_style=False)

    def headers(self):
        headers = {}
        token = self.get()
        if token:
            headers['Authorization'] = '%s %s' % (self.token_type, self.get())
        return headers