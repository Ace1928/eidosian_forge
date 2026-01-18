import os
import requests
from keystoneauth1 import loading
from keystoneauth1 import plugin
from oslo_log import log
def _authenticate_keycloak(self):
    keycloak_endpoint = '%s/realms/%s/protocol/openid-connect/token' % (self.auth_url, self.realm_name)
    body = {'grant_type': 'password', 'username': self.username, 'password': self.password, 'client_id': self.client_id, 'scope': 'profile'}
    resp = requests.post(keycloak_endpoint, data=body, verify=self.verify)
    try:
        resp.raise_for_status()
    except Exception as e:
        LOG.error('Failed to get access token: %s', str(e))
    return resp.json()['access_token']