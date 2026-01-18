import json
import logging
import os
import random
from .. import auth
from .. import constants
from .. import errors
from .. import utils
def _set_auth_headers(self, headers):
    log.debug('Looking for auth config')
    if not self._auth_configs or self._auth_configs.is_empty:
        log.debug('No auth config in memory - loading from filesystem')
        self._auth_configs = auth.load_config(credstore_env=self.credstore_env)
    if self._auth_configs:
        auth_data = self._auth_configs.get_all_credentials()
        if auth.INDEX_URL not in auth_data and auth.INDEX_NAME in auth_data:
            auth_data[auth.INDEX_URL] = auth_data.get(auth.INDEX_NAME, {})
        log.debug('Sending auth config ({})'.format(', '.join((repr(k) for k in auth_data.keys()))))
        if auth_data:
            headers['X-Registry-Config'] = auth.encode_header(auth_data)
    else:
        log.debug('No auth config found')