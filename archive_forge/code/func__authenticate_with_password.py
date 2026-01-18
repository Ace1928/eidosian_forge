import logging
import os
import pprint
import urllib
import requests
from mistralclient import auth
@staticmethod
def _authenticate_with_password(auth_url, client_id, client_secret, realm_name, username, password, cacert=None, insecure=None):
    access_token_endpoint = '%s/realms/%s/protocol/openid-connect/token' % (auth_url, realm_name)
    verify = None
    if urllib.parse.urlparse(access_token_endpoint).scheme == 'https':
        verify = False if insecure else cacert if cacert else True
    body = {'grant_type': 'password', 'username': username, 'password': password, 'client_id': client_id, 'scope': 'profile'}
    if client_secret:
        body['client_secret'] = (client_secret,)
    resp = requests.post(access_token_endpoint, data=body, verify=verify)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise Exception('Failed to get access token:\n %s' % str(e))
    LOG.debug('HTTP response from OIDC provider: %s', pprint.pformat(resp.json()))
    return resp.json()['access_token']