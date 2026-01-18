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
class KeycloakToken(object):
    """A token granted by a Keycloak server.

    Like sso.redhat.com as used by cloud.redhat.com
    ie Automation Hub"""
    token_type = 'Bearer'

    def __init__(self, access_token=None, auth_url=None, validate_certs=True, client_id=None):
        self.access_token = access_token
        self.auth_url = auth_url
        self._token = None
        self.validate_certs = validate_certs
        self.client_id = client_id
        if self.client_id is None:
            self.client_id = 'cloud-services'

    def _form_payload(self):
        return 'grant_type=refresh_token&client_id=%s&refresh_token=%s' % (self.client_id, self.access_token)

    def get(self):
        if self._token:
            return self._token
        payload = self._form_payload()
        resp = open_url(to_native(self.auth_url), data=payload, validate_certs=self.validate_certs, method='POST', http_agent=user_agent())
        data = json.loads(to_text(resp.read(), errors='surrogate_or_strict'))
        self._token = data.get('access_token')
        return self._token

    def headers(self):
        headers = {}
        headers['Authorization'] = '%s %s' % (self.token_type, self.get())
        return headers